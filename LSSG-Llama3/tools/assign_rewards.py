import re
import json
import argparse
from tqdm import tqdm
from textblob import TextBlob
from copy import deepcopy

from utils import randomly_convert_game_history_to_query
import ast
# import nltk
# nltk.download('all')

# PREDICT_TEMP = r"i know the word! it.{1,8}"
PREDICT_TEMP = """{"action": "accept"}"""

def get_derivative_words(word: str):
    # fuzzy matching for similar words
    word = word.lower()
    blob_word = TextBlob(word)
    word_list = [word, word + 'ing', word + 'ed', blob_word.words.pluralize()[0]]
    quotation_list = ["\"{word}\"", "'{word}'", '`{word}`']
    word_list += [quotation.format(word=word) for quotation in quotation_list for word in word_list]
    
    return word_list

def has_accepted(content: str):
    if re.search(PREDICT_TEMP, content.lower()):
        return True
    else:
        return False

def extract_actions(content):
    """
    从一条消息中提取所有结构化动作。
    支持：
    1) 整条就是 dict
    2) 文本里夹带一个或多个 dict
    """
    content = (content or "").strip()
    actions = []
    # 先尝试整条解析
    try:
        obj = ast.literal_eval(content)
        if isinstance(obj, dict) and "action" in obj:
            actions.append(obj)
            return actions
    except:
        pass
    # 再提取文本中的 {...} 片段
    candidates = re.findall(r"\{[^{}]*\}", content)
    for cand in candidates:
        try:
            obj = ast.literal_eval(cand)
            if isinstance(obj, dict) and "action" in obj:
                actions.append(obj)
        except:
            continue
    return actions


def get_game_outcome(history):
    for i, msg in enumerate(history):
        actions = extract_actions(msg.get("content", ""))
        for action_obj in actions:
            action = action_obj.get("action")
            if action == "accept":
                return "all win", i + 1
            if action == "quit":
                return "all lose", i + 1
    return "all lose", len(history)

def extract_nl_offer_price(content):
    """
    从自然语言中提取报价金额。
    只在明显像“出价/报价/还价”的句子里提取价格，
    避免把商品原价、尺寸、型号误识别成成交价。
    """
    text = (content or "").strip().lower()
    if not text:
        return None

    # 只有出现这些报价触发词时，才尝试抽价格
    trigger_patterns = [
        r"\boffer\b",
        r"\bpay\b",
        r"\btake\b",
        r"\baccept\b",
        r"\bgive\b",
        r"\bwilling to pay\b",
        r"\bwould pay\b",
        r"\bhow about\b",
        r"\bcan do\b",
        r"\bfor\s+\$?\d",
    ]

    if not any(re.search(p, text) for p in trigger_patterns):
        return None

    # 提取金额，支持 $295, 44.0, 1,200 等
    price_matches = re.findall(r"\$?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)", text)
    if not price_matches:
        return None

    # 默认取最后一个金额，更接近最终报价表达
    try:
        return float(price_matches[-1].replace(",", ""))
    except:
        return None
    

def extract_price_from_history(history):
    """
    提取最终成交价：
    1) 优先使用 accept 前最近一次结构化 offer
    2) 若无结构化 offer，则回退到 accept 前最近一次自然语言报价
    """
    for i in range(len(history) - 1, -1, -1):
        actions = extract_actions(history[i].get("content", ""))
        for act in actions:
            if act.get("action") == "accept":
                # 第一优先级：结构化 offer
                for j in range(i - 1, -1, -1):
                    prev_content = history[j].get("content", "")
                    prev_actions = extract_actions(prev_content)
                    for prev_act in reversed(prev_actions):
                        if prev_act.get("action") == "offer" and "price" in prev_act:
                            try:
                                return float(prev_act["price"])
                            except:
                                pass

                # 第二优先级：自然语言报价
                for j in range(i - 1, -1, -1):
                    prev_content = history[j].get("content", "")
                    nl_price = extract_nl_offer_price(prev_content)
                    if nl_price is not None:
                        return nl_price

                return None
    return None

def compute_terminal_utilities(original_price, deal_price, outcome, zeta_b=1.0, zeta_s=1.0):
    """
    根据论文中的定义计算 buyer / seller 的终局效用
    """
    if outcome != "all win" or deal_price is None or original_price <= 0:
        return 0.0, 0.0

    ratio = deal_price / original_price
    buyer_utility = zeta_b * (1.0 - ratio)
    seller_utility = zeta_s * ratio
    return buyer_utility, seller_utility

def compute_self_play_sample_rewards(game_episodes, input_data_path="", gamma=0.8, zeta_b=1.0, zeta_s=1.0):
    outputs = []
    judged_games = []
    buyer_game_num, seller_game_num = 0, 0

    for item in tqdm(game_episodes):
        history = item["history"]
        outcome, history_length = get_game_outcome(history)

        new_item = deepcopy(item)
        new_item["outcome"] = outcome
        judged_games.append(new_item)

        original_price = item.get("price", 0)
        deal_price = extract_price_from_history(history)

        buyer_terminal_reward, seller_terminal_reward = compute_terminal_utilities(
            original_price, deal_price, outcome, zeta_b=zeta_b, zeta_s=zeta_s
        )

        for i in range(history_length):
            message = history[i]
            content = message.get("content", "").strip()
            role = message["role"]

            query = randomly_convert_game_history_to_query(
                history[:i],
                item=item["item"],
                price=original_price,
                max_turns=item["max_turns"]
            )

            # discount to go
            steps_to_end = history_length - 1 - i
            discount = gamma ** steps_to_end

            if role == "buyer":
                reward = discount * buyer_terminal_reward
                buyer_game_num += 1
            else:
                reward = discount * seller_terminal_reward
                seller_game_num += 1

            outputs.append({
                "query": query,
                "target": content,
                "reward": reward,
                "role": role,
                "type": "rl"
            })

    json.dump(
        judged_games,
        open(input_data_path.replace(".json", "_judged.json"), "w"),
        ensure_ascii=False,
        indent=4,
    )

    all_game_num = buyer_game_num + seller_game_num
    buyer_weight = all_game_num / (2 * buyer_game_num) if buyer_game_num else 0.0
    seller_weight = all_game_num / (2 * seller_game_num) if seller_game_num else 0.0

    print(f"Processed {all_game_num} turns, buyer_weight={buyer_weight}, seller_weight={seller_weight}")

    for x in outputs:
        x["weight"] = buyer_weight if x["role"] == "buyer" else seller_weight

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser for episode processing.")
    parser.add_argument(
        "--input_data_path", type=str, default="", help="the path to input data."
    )
    parser.add_argument(
        "--output_data_path", type=str, default="", help="the path to output data."
    )
    parser.add_argument(
        "--sft_data_path", type=str, default="", help="the path to load sft data."
    )
    parser.add_argument(
        "--decay_weight", type=float, default=0.8, help="the decay weight of reward."
    )

    args = parser.parse_args()

    with open(args.input_data_path, "r") as f:
        game_episodes = json.load(f)

    results = compute_self_play_sample_rewards(
        game_episodes, args.input_data_path,  gamma=args.decay_weight
    )

    if args.sft_data_path:
        with open(args.sft_data_path, "r") as f:
            sft_data = json.load(f)
    else:
        sft_data = []

    for item in sft_data:
        item["type"] = "sft"
        item["weight"] = 1.0

    with open(args.output_data_path, "w") as f:
        json.dump(results + sft_data, f, ensure_ascii=False, indent=2)
