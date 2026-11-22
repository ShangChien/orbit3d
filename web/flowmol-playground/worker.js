let flowmolMod = null;
let uni3darMod = null;
let dpa4Mod = null;
let hyffMod = null;

// Decoupled dual-slot architecture:
// - Slot 1: Generative 3D Engine (`FlowMol-118` or `Uni-3DAR-Mol`)
// - Slot 2: Quantum MLIP Force-Field & MD Engine (`DPA-4` or `HyFF-L2`)
let genEngine = null;
let genFamily = null; // "flowmol" | "uni3dar"

let mlipEngine = null;
let mlipFamily = null; // "dpa4" | "hyff"

const MAGIC_FM118 = [70, 77, 49, 49, 56, 1, 0, 0];       // b"FM118\x01\x00\x00"
const MAGIC_U3DM  = [85, 51, 68, 77, 1, 0, 0, 0];        // b"U3DM\x01\x00\x00\x00"
const MAGIC_DPA4  = [68, 80, 65, 52, 82, 83, 48, 49];    // b"DPA4RS01"
const MAGIC_HYFF  = [72, 89, 70, 70, 82, 83, 48, 49];    // b"HYFFRS01"

function detectBundleKind(u8) {
  if (!u8 || u8.byteLength < 64) return null;
  if (MAGIC_FM118.every((b, i) => u8[i] === b)) return "flowmol";
  if (MAGIC_U3DM.every((b, i) => u8[i] === b)) return "uni3dar";
  if (MAGIC_DPA4.every((b, i) => u8[i] === b)) return "dpa4";
  if (MAGIC_HYFF.every((b, i) => u8[i] === b)) return "hyff";
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
    const m = await import("./pkg/flowmol_rs.js?v=7");
    await m.default("./pkg/flowmol_rs_bg.wasm?v=7");
    flowmolMod = m;
  }
  return flowmolMod;
}

async function ensureUni3dar() {
  if (!uni3darMod) {
    const m = await import("./pkg_uni3dar/uni3dar_rs.js?v=7");
    await m.default("./pkg_uni3dar/uni3dar_rs_bg.wasm?v=7");
    uni3darMod = m;
  }
  return uni3darMod;
}

async function ensureDpa4() {
  if (!dpa4Mod) {
    const m = await import("./pkg_dpa4/dpa4_rs.js?v=7");
    await m.default("./pkg_dpa4/dpa4_rs_bg.wasm?v=7");
    dpa4Mod = m;
  }
  return dpa4Mod;
}

async function ensureHyff() {
  if (!hyffMod) {
    const m = await import("./pkg_hyff/hyff_rs.js?v=7");
    await m.default("./pkg_hyff/hyff_rs_bg.wasm?v=7");
    hyffMod = m;
  }
  return hyffMod;
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

async function fetchOrCache(url, cacheKey, bypassCache, id) {
  let u8 = null;
  if (!bypassCache && self.navigator?.storage?.getDirectory) {
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
    self.postMessage({ type: "status", id, text: `正在下载 ${url}...` });
    u8 = await fetchWithProgress(url, id);
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
  return u8;
}

async function loadFromU8(u8, label, useGpu, id) {
  const caps = getEnvCaps();
  const kind = detectBundleKind(u8);
  if (!kind) {
    throw new Error(
      `文件 "${label}" 不是有效的 .fm118 / .u3m / .dpa4 / .hyff 模型包`
    );
  }

  const wantGpu = useGpu !== false && caps.hasWebGPU;
  const t0 = performance.now();

  if (kind === "dpa4" || kind === "hyff") {
    if (mlipEngine) {
      try { mlipEngine.free(); } catch (_) {}
      mlipEngine = null;
      mlipFamily = null;
    }
    self.postMessage({
      type: "status",
      id,
      text: `正在初始化 ${kind === "dpa4" ? "DPA-4 (SeZM)" : "HyFF-L2"} 量子势场管线 (${(u8.byteLength / 1e6).toFixed(1)} MB)...`,
    });

    if (kind === "dpa4") {
      const m = await ensureDpa4();
      mlipEngine = await m.WasmDpa4Engine.create(u8, wantGpu);
    } else {
      const m = await ensureHyff();
      mlipEngine = await m.WasmHyffEngine.create(u8, wantGpu);
    }
    mlipFamily = kind;
    const backendStr = mlipEngine.backendInfo();
    const hasOracle = kind === "dpa4" && typeof mlipEngine.hasPropertyOracle === "function"
      ? mlipEngine.hasPropertyOracle()
      : false;

    self.postMessage({
      type: "mlip_loaded",
      id,
      label,
      info: {
        mlip_family: kind,
        backend: backendStr,
        is_gpu: backendStr.includes("WebGPU"),
        has_oracle: hasOracle,
        load_ms: performance.now() - t0,
        size_mb: +(u8.byteLength / 1e6).toFixed(2),
      },
    });
    return;
  }

  // Generative slot ("flowmol" | "uni3dar")
  if (genEngine) {
    try { genEngine.free(); } catch (_) {}
    genEngine = null;
    genFamily = null;
  }
  self.postMessage({
    type: "status",
    id,
    text: `正在初始化 ${kind === "uni3dar" ? "Uni-3DAR-Mol" : "FlowMol-118"} WebGPU 管线 (${(u8.byteLength / 1e6).toFixed(1)} MB)...`,
  });

  if (kind === "uni3dar") {
    const m = await ensureUni3dar();
    genEngine = await m.WasmUni3darEngine.create(u8, wantGpu);
  } else {
    const m = await ensureFlowMol();
    genEngine = await m.WasmFlowMolEngine.create(u8, wantGpu);
  }
  genFamily = kind;

  const info = JSON.parse(genEngine.backendInfo());
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
      const u8 = await fetchOrCache(msg.url, cacheKey, msg.bypassCache, msg.id);
      const label = msg.url.split("/").pop() || "model";
      await loadFromU8(u8, label, msg.useGpu, msg.id);
      return;
    }

    if (msg.type === "sample") {
      if (!genEngine) {
        throw new Error("请先加载 .fm118 或 .u3m 生成模型文件");
      }
      const t0 = performance.now();
      const resStr = await genEngine.sampleMolecule(msg.optsJson, (step, total, xFlat, aIdx) => {
        const xCopy = new Float32Array(xFlat);
        const aCopy = new Uint32Array(aIdx);
        self.postMessage(
          { type: "step", id: msg.id, engineFamily: genFamily, step, total, xFlat: xCopy, aIdx: aCopy },
          [xCopy.buffer, aCopy.buffer]
        );
      });
      const dt = performance.now() - t0;
      self.postMessage({
        type: "sampled",
        id: msg.id,
        engineFamily: genFamily,
        resStr,
        elapsedMs: dt,
        autoRelax: !!msg.autoRelax,
      });
      return;
    }

    if (msg.type === "mlip_eval") {
      if (!mlipEngine) {
        throw new Error("请先在下方势场面板加载 DPA-4 (.dpa4) 或 HyFF-L2 (.hyff) 权重");
      }
      const zArr = new Uint8Array(msg.atomicNumbers);
      const cArr = new Float64Array(msg.coordsFlat);
      const t0 = performance.now();
      const resStr = await mlipEngine.evalSingle(
        zArr,
        cArr,
        msg.charge ?? 0.0,
        msg.multiplicity ?? 1,
        true,
        false
      );
      self.postMessage({
        type: "mlip_eval_done",
        id: msg.id,
        mlipFamily,
        resStr,
        elapsedMs: performance.now() - t0,
      });
      return;
    }

    if (msg.type === "mlip_relax") {
      if (!mlipEngine) {
        throw new Error("请先在下方势场面板加载 DPA-4 (.dpa4) 或 HyFF-L2 (.hyff) 权重");
      }
      const zArr = new Uint8Array(msg.atomicNumbers);
      const cArr = new Float64Array(msg.coordsFlat);
      const t0 = performance.now();
      let lastPost = 0;
      const resStr = await mlipEngine.relaxGeometry(
        zArr,
        cArr,
        msg.charge ?? 0.0,
        msg.multiplicity ?? 1,
        msg.optsJson || "",
        (frameStr) => {
          const now = performance.now();
          if (now - lastPost >= 24) {
            lastPost = now;
            self.postMessage({ type: "mlip_relax_frame", id: msg.id, frameStr });
          }
        }
      );
      self.postMessage({
        type: "mlip_relax_done",
        id: msg.id,
        mlipFamily,
        resStr,
        elapsedMs: performance.now() - t0,
      });
      return;
    }

    if (msg.type === "mlip_md") {
      if (!mlipEngine) {
        throw new Error("请先在下方势场面板加载 DPA-4 (.dpa4) 或 HyFF-L2 (.hyff) 权重");
      }
      const zArr = new Uint8Array(msg.atomicNumbers);
      const cArr = new Float64Array(msg.coordsFlat);
      const t0 = performance.now();
      let lastPost = 0;
      const resStr = await mlipEngine.runMd(
        zArr,
        cArr,
        msg.charge ?? 0.0,
        msg.multiplicity ?? 1,
        msg.optsJson || "",
        (frameStr) => {
          const now = performance.now();
          if (now - lastPost >= 24) {
            lastPost = now;
            self.postMessage({ type: "mlip_md_frame", id: msg.id, frameStr });
          }
        }
      );
      self.postMessage({
        type: "mlip_md_done",
        id: msg.id,
        mlipFamily,
        resStr,
        elapsedMs: performance.now() - t0,
      });
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
