from typing import Any, Optional


def find_provider(context: Any, provider_id: str) -> Optional[Any]:
    """Find provider by id/provider_id/name across LLM and STT provider pools."""
    if not provider_id:
        return None

    all_lists = []
    try:
        all_lists.append(context.get_all_providers())
    except Exception:
        pass
    try:
        all_lists.append(context.get_all_stt_providers())
    except Exception:
        pass

    for p_list in all_lists:
        if not p_list:
            continue
        for provider in p_list:
            candidates = set()
            for attr in ("id", "provider_id", "name"):
                val = getattr(provider, attr, None)
                if val:
                    candidates.add(str(val))

            cfg = getattr(provider, "provider_config", None)
            if isinstance(cfg, dict):
                for key in ("id", "provider_id", "name"):
                    val = cfg.get(key)
                    if val:
                        candidates.add(str(val))

            if provider_id in candidates:
                return provider

    return None
