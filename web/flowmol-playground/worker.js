let flowmolMod = null;
let uni3darMod = null;
let engine = null;
let engineFamily = null; // "flowmol" | "uni3dar"

const MAGIC_FM118 = [70, 77, 49, 49, 56, 1, 0, 0]; // b"FM118\x01\x00\x00"
const MAGIC_U3DM  = [85, 51, 68, 77, 1, 0, 0, 0];  // b"U3DM\x01\x00\x00\x00"

function detectBundleKind(u8) {
  if (!u8 || u8.byteLength < 64) return null;
  if (MAGIC_FM118.every((b, i) => u8[i] === b)) return "flowmol";
  if (MAGIC_U3DM.every((b, i) => u8[i] === b)) return "uni3dar";
  return null;
}

function getEnvCaps() {
  return {
    hasWebGPU: typeof navigator !== "undefined" && !!navigator.gpu,
    isSecure: typeof self.isSecureContext === "boolean" ? self.isSecureContext : true,
  };
}

async function ensureFlowMol() {
  if (!flowmolMod) {
    const m = await import("./pkg/flowmol_rs.js?v=6");
    await m.default("./pkg/flowmol_rs_bg.wasm?v=6");
    flowmolMod = m;
  }
  return flowmolMod;
}

async function ensureUni3dar() {
  if (!uni3darMod) {
    const m = await import("./pkg_uni3dar/uni3dar_rs.js?v=6");
    await m.default("./pkg_uni3dar/uni3dar_rs_bg.wasm?v=6");
    uni3darMod = m;
  }
  return uni3darMod;
}

async function fetchWithProgress(url, id) {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status} (${url})`);
  }
  const total = parseInt(resp.headers.get("content-length") || "0", 10);
  if (!resp.body || !total) {
    return new Uint8Array(await resp.arrayBuffer());
  }
  const out = new Uint8Array(total);
  const reader = resp.body.getReader();
  let offset = 0;
  let lastPost = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (offset + value.byteLength > out.byteLength) {
      const next = new Uint8Array(offset + value.byteLength);
      next.set(out.subarray(0, offset));
      next.set(value, offset);
      return next;
    }
    out.set(value, offset);
    offset += value.byteLength;
    const now = performance.now();
    if (now - lastPost > 80 || offset === total) {
      lastPost = now;
      self.postMessage({ type: "download_progress", id, loaded: offset, total });
    }
  }
  return offset === out.byteLength ? out : out.subarray(0, offset);
}

async function loadFromU8(u8, label, useGpu, id) {
  const caps = getEnvCaps();
  const kind = detectBundleKind(u8);
  if (!kind) {
    throw new Error(
      `文件 "${label}" 不是有效的 .fm118 (FlowMol-118) 或 .u3m (Uni-3DAR-Mol) 模型包`
    );
  }
  if (engine) {
    try { engine.free(); } catch (_) {}
    engine = null;
    engineFamily = null;
  }
  self.postMessage({
    type: "status",
    id,
    text: `正在初始化 ${kind === "uni3dar" ? "Uni-3DAR-Mol" : "FlowMol-118"} WebGPU 管线 (${(u8.byteLength / 1e6).toFixed(1)} MB)...`,
  });

  const wantGpu = useGpu !== false && caps.hasWebGPU;
  const t0 = performance.now();
  if (kind === "uni3dar") {
    const m = await ensureUni3dar();
    engine = await m.WasmUni3darEngine.create(u8, wantGpu);
  } else {
    const m = await ensureFlowMol();
    engine = await m.WasmFlowMolEngine.create(u8, wantGpu);
  }
  engineFamily = kind;

  const info = JSON.parse(engine.backendInfo());
  info.engine_family = kind;
  info.load_ms = performance.now() - t0;
  info.size_mb = +(u8.byteLength / 1e6).toFixed(2);
  if (!caps.hasWebGPU && !info.gpu_error) {
    info.gpu_error = caps.isSecure
      ? "当前浏览器未开启 WebGPU (navigator.gpu 不可用)"
      : "当前非安全上下文 (请使用 http://localhost:8080 或 http://127.0.0.1:8080 访问以解锁 WebGPU)";
  }
  self.postMessage({ type: "loaded", id, label, info });
}

self.onmessage = async (ev) => {
  const msg = ev.data || {};
  try {
    if (msg.type === "init") {
      self.postMessage({ type: "inited", ...getEnvCaps() });
      return;
    }

    if (msg.type === "load_file") {
      self.postMessage({ type: "status", id: msg.id, text: `正在读取本地文件 ${msg.file.name}...` });
      const u8 = new Uint8Array(await msg.file.arrayBuffer());
      await loadFromU8(u8, msg.file.name, msg.useGpu, msg.id);
      return;
    }

    if (msg.type === "load_url") {
      const cacheKey = msg.cacheKey || msg.url.replace(/[^a-zA-Z0-9._-]/g, "_");
      let u8 = null;
      if (!msg.bypassCache && self.navigator?.storage?.getDirectory) {
        try {
          const root = await self.navigator.storage.getDirectory();
          const fh = await root.getFileHandle(cacheKey);
          const candidate = new Uint8Array(await (await fh.getFile()).arrayBuffer());
          if (detectBundleKind(candidate)) {
            u8 = candidate;
          }
        } catch (_) {}
      }
      if (!u8) {
        self.postMessage({ type: "status", id: msg.id, text: `正在下载 ${msg.url}...` });
        u8 = await fetchWithProgress(msg.url, msg.id);
        if (detectBundleKind(u8) && self.navigator?.storage?.getDirectory) {
          try {
            const root = await self.navigator.storage.getDirectory();
            const fh = await root.getFileHandle(cacheKey, { create: true });
            const w = await fh.createWritable();
            await w.write(u8);
            await w.close();
          } catch (_) {}
        }
      }
      const label = msg.url.split("/").pop() || "model";
      await loadFromU8(u8, label, msg.useGpu, msg.id);
      return;
    }

    if (msg.type === "sample") {
      if (!engine) {
        throw new Error("请先加载 .fm118 或 .u3m 模型文件");
      }
      const t0 = performance.now();
      const resStr = await engine.sampleMolecule(msg.optsJson, (step, total, xFlat, aIdx) => {
        const xCopy = new Float32Array(xFlat);
        const aCopy = new Uint32Array(aIdx);
        self.postMessage(
          { type: "step", id: msg.id, engineFamily, step, total, xFlat: xCopy, aIdx: aCopy },
          [xCopy.buffer, aCopy.buffer]
        );
      });
      const dt = performance.now() - t0;
      self.postMessage({ type: "sampled", id: msg.id, engineFamily, resStr, elapsedMs: dt });
      return;
    }
  } catch (err) {
    self.postMessage({
      type: "error",
      id: msg.id,
      error: err && err.message ? err.message : String(err),
    });
  }
};
