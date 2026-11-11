from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from . import function as function
from . import function as fn

__version__ = "2.4.0-compat"


class _IndexedNodeView:
    def __init__(self, ndata: Dict[str, torch.Tensor], idx: torch.Tensor):
        self._ndata = ndata
        self._idx = idx

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._ndata[key][self._idx]


class EdgeBatch:
    def __init__(self, g: "DGLGraph"):
        self._g = g
        self.src = _IndexedNodeView(g.ndata, g._src)
        self.dst = _IndexedNodeView(g.ndata, g._dst)
        self.data = g.edata


class DGLGraph:
    """Pure-PyTorch Differentiable Graph compatible with PropMolFlow's DGLGraph usage."""

    def __init__(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        batch_num_nodes: Optional[torch.Tensor] = None,
        batch_num_edges: Optional[torch.Tensor] = None,
    ):
        if device is None:
            device = src.device if isinstance(src, torch.Tensor) else torch.device("cpu")
        self._device = torch.device(device)
        self._src = torch.as_tensor(src, dtype=torch.long, device=self._device)
        self._dst = torch.as_tensor(dst, dtype=torch.long, device=self._device)
        if num_nodes is None:
            if self._src.numel() > 0:
                num_nodes = int(max(self._src.max().item(), self._dst.max().item()) + 1)
            else:
                num_nodes = 0
        self._num_nodes = int(num_nodes)
        if batch_num_nodes is None:
            self._batch_num_nodes = torch.tensor([self._num_nodes], dtype=torch.long, device=self._device)
        else:
            self._batch_num_nodes = torch.as_tensor(batch_num_nodes, dtype=torch.long, device=self._device)
        if batch_num_edges is None:
            self._batch_num_edges = torch.tensor([self._src.numel()], dtype=torch.long, device=self._device)
        else:
            self._batch_num_edges = torch.as_tensor(batch_num_edges, dtype=torch.long, device=self._device)

        self.ndata: Dict[str, torch.Tensor] = {}
        self.edata: Dict[str, torch.Tensor] = {}

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def batch_size(self) -> int:
        return int(self._batch_num_nodes.numel())

    def num_nodes(self) -> int:
        return self._num_nodes

    def num_edges(self) -> int:
        return int(self._src.numel())

    def batch_num_nodes(self) -> torch.Tensor:
        return self._batch_num_nodes

    def batch_num_edges(self) -> torch.Tensor:
        return self._batch_num_edges

    def edges(self, form: str = "uv") -> Tuple[torch.Tensor, torch.Tensor]:
        if form == "uv":
            return self._src, self._dst
        elif form == "eid":
            return torch.arange(self.num_edges(), device=self._device, dtype=torch.long)
        elif form == "all":
            return (
                self._src,
                self._dst,
                torch.arange(self.num_edges(), device=self._device, dtype=torch.long),
            )
        raise ValueError(f"Unsupported form={form}")

    def to(self, device: Union[str, torch.device]) -> "DGLGraph":
        target = torch.device(device)
        new_g = DGLGraph(
            self._src.to(target),
            self._dst.to(target),
            num_nodes=self._num_nodes,
            device=target,
            batch_num_nodes=self._batch_num_nodes.to(target),
            batch_num_edges=self._batch_num_edges.to(target),
        )
        for k, v in self.ndata.items():
            new_g.ndata[k] = v.to(target)
        for k, v in self.edata.items():
            new_g.edata[k] = v.to(target)
        return new_g

    @contextmanager
    def local_scope(self):
        saved_ndata = dict(self.ndata)
        saved_edata = dict(self.edata)
        try:
            yield
        finally:
            self.ndata = saved_ndata
            self.edata = saved_edata

    def apply_edges(self, func: Union[function.EdgeOp, Callable]):
        if isinstance(func, function.EdgeOp):
            if func.op == "u_sub_v":
                self.edata[func.out] = self.ndata[func.lhs][self._src] - self.ndata[func.rhs][self._dst]
            elif func.op == "u_add_v":
                self.edata[func.out] = self.ndata[func.lhs][self._src] + self.ndata[func.rhs][self._dst]
            elif func.op == "copy_e":
                self.edata[func.out] = self.edata[func.lhs]
            elif func.op == "copy_u":
                self.edata[func.out] = self.ndata[func.lhs][self._src]
            else:
                raise NotImplementedError(f"Unsupported EdgeOp: {func}")
        else:
            eb = EdgeBatch(self)
            out = func(eb)
            if isinstance(out, dict):
                self.edata.update(out)

    def update_all(
        self,
        message_func: Union[function.EdgeOp, Callable],
        reduce_func: function.ReduceOp,
    ):
        if isinstance(message_func, function.EdgeOp):
            if message_func.op == "copy_e":
                msg = self.edata[message_func.lhs]
            elif message_func.op == "copy_u":
                msg = self.ndata[message_func.lhs][self._src]
            elif message_func.op == "u_sub_v":
                msg = self.ndata[message_func.lhs][self._src] - self.ndata[message_func.rhs][self._dst]
            else:
                raise NotImplementedError(f"Unsupported message_func: {message_func}")
        else:
            eb = EdgeBatch(self)
            res = message_func(eb)
            msg = res[reduce_func.msg]

        out_shape = (self._num_nodes,) + msg.shape[1:]
        agg = torch.zeros(out_shape, dtype=msg.dtype, device=msg.device)
        agg.index_add_(0, self._dst, msg)

        if reduce_func.op == "mean":
            ones = torch.ones(self._dst.numel(), dtype=msg.dtype, device=msg.device)
            deg = torch.zeros(self._num_nodes, dtype=msg.dtype, device=msg.device)
            deg.index_add_(0, self._dst, ones)
            deg = deg.clamp(min=1.0)
            view_shape = (self._num_nodes,) + (1,) * (msg.ndim - 1)
            agg = agg / deg.view(view_shape)
        elif reduce_func.op != "sum":
            raise NotImplementedError(f"Unsupported reduce_func: {reduce_func}")

        self.ndata[reduce_func.out] = agg


def graph(
    data: Tuple[Union[torch.Tensor, List[int]], Union[torch.Tensor, List[int]]],
    num_nodes: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> DGLGraph:
    src, dst = data
    return DGLGraph(src, dst, num_nodes=num_nodes, device=device)


def batch(graphs: List[DGLGraph]) -> DGLGraph:
    if len(graphs) == 0:
        raise ValueError("Cannot batch an empty list of graphs")
    device = graphs[0].device
    src_list = []
    dst_list = []
    b_nodes = []
    b_edges = []
    node_offset = 0
    for g in graphs:
        src_list.append(g._src + node_offset)
        dst_list.append(g._dst + node_offset)
        b_nodes.append(g.batch_num_nodes())
        b_edges.append(g.batch_num_edges())
        node_offset += g.num_nodes()

    batched = DGLGraph(
        src=torch.cat(src_list, dim=0) if src_list else torch.empty(0, dtype=torch.long, device=device),
        dst=torch.cat(dst_list, dim=0) if dst_list else torch.empty(0, dtype=torch.long, device=device),
        num_nodes=node_offset,
        device=device,
        batch_num_nodes=torch.cat(b_nodes, dim=0),
        batch_num_edges=torch.cat(b_edges, dim=0),
    )

    for k in graphs[0].ndata.keys():
        batched.ndata[k] = torch.cat([g.ndata[k] for g in graphs], dim=0)
    for k in graphs[0].edata.keys():
        batched.edata[k] = torch.cat([g.edata[k] for g in graphs], dim=0)
    return batched


def unbatch(g: DGLGraph) -> List[DGLGraph]:
    b_nodes = g.batch_num_nodes().detach().cpu().tolist()
    b_edges = g.batch_num_edges().detach().cpu().tolist()

    ndata_splits = {k: torch.split(v, b_nodes, dim=0) for k, v in g.ndata.items()}
    edata_splits = {k: torch.split(v, b_edges, dim=0) for k, v in g.edata.items()}
    src_splits = torch.split(g._src, b_edges, dim=0)
    dst_splits = torch.split(g._dst, b_edges, dim=0)

    out_graphs = []
    node_offset = 0
    for i, (n_i, e_i) in enumerate(zip(b_nodes, b_edges)):
        sub_g = DGLGraph(
            src=src_splits[i] - node_offset,
            dst=dst_splits[i] - node_offset,
            num_nodes=n_i,
            device=g.device,
        )
        for k, splits in ndata_splits.items():
            sub_g.ndata[k] = splits[i]
        for k, splits in edata_splits.items():
            sub_g.edata[k] = splits[i]
        out_graphs.append(sub_g)
        node_offset += n_i
    return out_graphs


def readout_nodes(
    g: DGLGraph,
    feat: str,
    op: str = "sum",
    weight: Optional[str] = None,
) -> torch.Tensor:
    h = g.ndata[feat]
    if weight is not None:
        w = g.ndata[weight]
        while w.ndim < h.ndim:
            w = w.unsqueeze(-1)
        h = h * w

    b_nodes = g.batch_num_nodes()
    batch_size = g.batch_size
    batch_idx = torch.repeat_interleave(
        torch.arange(batch_size, device=h.device, dtype=torch.long),
        b_nodes,
    )
    out_shape = (batch_size,) + h.shape[1:]
    out = torch.zeros(out_shape, dtype=h.dtype, device=h.device)
    out.index_add_(0, batch_idx, h)

    if op == "mean":
        denom = b_nodes.clamp(min=1).to(h.dtype)
        view_shape = (batch_size,) + (1,) * (h.ndim - 1)
        out = out / denom.view(view_shape)
    elif op != "sum":
        raise NotImplementedError(f"Unsupported readout op={op}")
    return out
