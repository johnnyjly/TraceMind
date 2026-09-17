"""Retain diagnostic dimensions without exporting arbitrary payloads."""

from hashlib import sha256


DIAGNOSTIC_KEYS = frozenset(
    {
        "rpc.system.name",
        "rpc.response.status_code",
        "grpc.method",
        "grpc.status_code",
        "span.kind",
        "demo.order.items.count",
        "rpc.system",
        "rpc.service",
        "rpc.method",
        "rpc.grpc.status_code",
        "server.address",
        "server.port",
        "peer.service",
        "net.peer.name",
        "net.peer.port",
        "http.request.method",
        "http.method",
        "http.response.status_code",
        "http.status_code",
        "http.route",
        "db.system",
        "db.system.name",
        "db.namespace",
        "db.name",
        "db.operation",
        "db.operation.name",
        "db.collection.name",
        "db.statement",
        "db.query.text",
        "db.sql.table",
        "error.type",
        "otel.status_description",
        "status.message",
        "demo.cart.items.count",
        "demo.product.quantity",
        "cart.items.count",
        "cart.item.count",
        "cart.total",
        "items.count",
        "cart.operation",
        "app.cart.items.count",
        "app.cart.items.quantity",
        "feature_flag.key",
        "feature_flag.variant",
        "flag",
        "variant",
        "delay_secs",
        "country",
        "shipping.country",
        "operation",
    }
)


def diagnostic_attributes(values: dict) -> dict:
    selected = {}
    aliases = {
        "userId": "user.id",
        "user_id": "user.id",
        "feature_flag_key": "feature_flag.key",
        "feature_flag_variant": "feature_flag.variant",
        "quantity": "demo.product.quantity",
    }
    for key, value in values.items():
        key = aliases.get(key, key)
        if key == "user.id" and isinstance(value, (str, int)):
            selected["user.id.sha256"] = sha256(
                str(value).encode()
            ).hexdigest()
        elif key in DIAGNOSTIC_KEYS and isinstance(
            value, (str, int, float, bool)
        ):
            selected[key] = value[:1600] if isinstance(value, str) else value
            if isinstance(value, str) and len(value) > 1600:
                selected[key + ".truncated"] = True
    return selected
