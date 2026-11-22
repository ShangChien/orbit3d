import init, { WasmFlowMolEngine } from "./pkg/flowmol_rs.js?v=4";

let wasmReady = null;
let engine = null;

const MAGIC = [70, 77, 49, 49, 56, 1, 0, 0]; // b"FM118\x01\x00\x00"

function checkMagic(u8) {
  return u8 && u8.byteLength >= 64 && MAGIC.every((b, i) => u8[i] === b);
}

async function ensureInit() {
  if (!wasmReady) {
    wasmReady = init();
  }
  await wasmReady;
  return {
    hasWebGPU: typeof navigator !== "undefined" && !!navigator.gpu,
    isSecure: typeof self.isSecureContext === "boolean" ? self.isSecureContext : true,
  };
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
      // Fallback if Content-Length mismatched
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
  const caps = await ensureInit();
  if (!checkMagic(u8)) {
    throw new Error(
      `文件 "${label}" 不是有效的 .fm118 模型包（若选择了 PyTorch .pt 文件，请改选导出后的 *.fm118 文件）`
    );
  }
  if (engine) {
    try { engine.free(); } catch (_) {}
    engine = null;
  }
  self.postMessage({ type: "status", id, text: `正在初始化推理管线 (${(u8.byteLength / 1e6).toFixed(1)} MB)...` });
  const wantGpu = useGpu !== false && caps.hasWebGPU;
  const t0 = performance.now();
  engine = await WasmFlowMolEngine.create(u8, wantGpu);
  const info = JSON.parse(engine.backendInfo());
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
      const caps = await ensureInit();
      self.postMessage({ type: "inited", ...caps });
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
      let fromCache = false;
      if (!msg.bypassCache && self.navigator?.storage?.getDirectory) {
        try {
          const root = await self.navigator.storage.getDirectory();
          const fh = await root.getFileHandle(cacheKey);
          const candidate = new Uint8Array(await (await fh.getFile()).arrayBuffer());
          if (checkMagic(candidate)) {
            u8 = candidate;
            fromCache = true;
          }
        } catch (_) {}
      }
      if (!u8) {
        self.postMessage({ type: "status", id: msg.id, text: `正在下载 ${msg.url}...` });
        u8 = await fetchWithProgress(msg.url, msg.id);
        if (checkMagic(u8) && self.navigator?.storage?.getDirectory) {
          try {
            const root = await self.navigator.storage.getDirectory();
            const fh = await root.getFileHandle(cacheKey, { create: true });
            if (fh.createWritable) {
              const wr = await fh.createWritable();
              await wr.write(u8);
              await wr.close();
            }
          } catch (_) {}
        }
      }
      await loadFromU8(u8, fromCache ? `OPFS/${cacheKey}` : msg.url, msg.useGpu, msg.id);
      return;
    }

    if (msg.type === "sample") {
      if (!engine) {
        throw new Error("请先加载 .fm118 模型包");
      }
      const t0 = performance.now();
      const resStr = await engine.sampleMolecule(msg.optsJson, (step, total, xFlat, aIdx) => {
        const xCopy = new Float32Array(xFlat);
        const aCopy = new Uint32Array(aIdx);
        self.postMessage(
          { type: "step", id: msg.id, step, total, xFlat: xCopy, aIdx: aCopy },
          [xCopy.buffer, aCopy.buffer]
        );
      });
      const elapsedMs = performance.now() - t0;
      self.postMessage({ type: "sampled", id: msg.id, resStr, elapsedMs });
      return;
    }
  } catch (err) {
    const errMsg = err instanceof Error ? err.message : String(err);
    self.postMessage({ type: "error", id: msg.id, phase: msg.type, error: errMsg });
  }
};
