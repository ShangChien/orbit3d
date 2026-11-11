from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeOp:
    op: str
    lhs: str
    rhs: str = ""
    out: str = ""


@dataclass(frozen=True)
class ReduceOp:
    op: str
    msg: str
    out: str


def u_sub_v(lhs_field: str, rhs_field: str, out: str) -> EdgeOp:
    return EdgeOp(op="u_sub_v", lhs=lhs_field, rhs=rhs_field, out=out)


def u_add_v(lhs_field: str, rhs_field: str, out: str) -> EdgeOp:
    return EdgeOp(op="u_add_v", lhs=lhs_field, rhs=rhs_field, out=out)


def copy_e(field: str, out: str) -> EdgeOp:
    return EdgeOp(op="copy_e", lhs=field, out=out)


def copy_u(field: str, out: str) -> EdgeOp:
    return EdgeOp(op="copy_u", lhs=field, out=out)


def sum(msg: str, out: str) -> ReduceOp:
    return ReduceOp(op="sum", msg=msg, out=out)


def mean(msg: str, out: str) -> ReduceOp:
    return ReduceOp(op="mean", msg=msg, out=out)
