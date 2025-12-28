import json
from typing import List, Dict, Any, Tuple
from astrbot.api import logger

async def extract_content_recursively(message_nodes: List[Dict[str, Any]], extracted_texts: list[str], image_urls: list[str], depth: int = 0):
    """
    核心递归解析器。遍历消息节点列表，提取文本、图片，并处理嵌套的 forward 结构。
    """
    indent = "  " * depth
    for message_node in message_nodes: 
        sender_name = message_node.get("sender", {}).get("nickname", "未知用户") 
        raw_content = message_node.get("message") or message_node.get("content", []) 

        content_chain = [] 
        if isinstance(raw_content, str): 
            try: 
                parsed_content = json.loads(raw_content) 
                if isinstance(parsed_content, list): 
                    content_chain = parsed_content 
            except (json.JSONDecodeError, TypeError): 
                content_chain = [{"type": "text", "data": {"text": raw_content}}] 
        elif isinstance(raw_content, list): 
            content_chain = raw_content 

        node_text_parts = [] 
        has_only_forward = False
        
        if isinstance(content_chain, list):
            if len(content_chain) == 1 and content_chain[0].get("type") == "forward":
                has_only_forward = True
                
            for segment in content_chain: 
                if isinstance(segment, dict): 
                    seg_type = segment.get("type") 
                    seg_data = segment.get("data", {}) 
                    
                    if seg_type == "text": 
                        text = seg_data.get("text", "") 
                        if text: node_text_parts.append(text) 
                    elif seg_type == "image": 
                        url = seg_data.get("url") 
                        if url: 
                            image_urls.append(url) 
                            node_text_parts.append("[图片]") 
                    elif seg_type == "forward":
                        nested_content = seg_data.get("content")
                        if isinstance(nested_content, list):
                            await extract_content_recursively(nested_content, extracted_texts, image_urls, depth + 1)
                        else:
                            node_text_parts.append("[转发消息内容缺失或格式错误]")

        full_node_text = "".join(node_text_parts).strip()
        if full_node_text and not has_only_forward: 
            extracted_texts.append(f"{indent}{sender_name}: {full_node_text}")

async def extract_forward_content(client: Any, forward_id: str) -> Tuple[list[str], list[str]]:
    """
    从合并转发消息中提取内容。
    """
    extracted_texts = [] 
    image_urls = []
    try: 
        forward_data = await client.api.call_action('get_forward_msg', id=forward_id) 
    except Exception as e: 
        logger.warning(f"调用 get_forward_msg API 失败 (ID: {forward_id}): {e}") 
        return [], [] 

    if not forward_data or "messages" not in forward_data: 
        return [], [] 

    await extract_content_recursively(forward_data["messages"], extracted_texts, image_urls, depth=0)
    return extracted_texts, image_urls 
