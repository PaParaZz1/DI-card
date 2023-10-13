

# 假设我们拿到的是game_info

class SimpleBot():
    '''
    Overview: 
    '''
    def __init__(self, ):
        super(SimpleBot, self).__init__()
        self.characters = ['Klee', 'Xingqiu', 'Kaeya']    # Chongyun
        self.charge_cards = ['兽肉薄荷卷', '烟熏鸡', '刘苏']
        self.stockpile_cards = ['参量质变仪', '立本', '派蒙']
        self.optional_card = ['常九爷', '换班时间', '凯瑟琳', '交给我吧', '砰砰礼物']

    def choose_cards_to_replace(self, initial_hand):
        # 开局初始化
        mask = [1, 1, 1, 1, 1]  # 初始化mask，表示所有牌都可以替换

        def check_card_priority(card, check_list):
            if card in check_list:
                return check_list.index(card)
            else:
                return -1

        # 检查是否有充能牌，按优先级选择留下1张
        card_priority = [check_card_priority(card, self.charge_cards) for card in initial_hand]
        sorted_priority = sorted(card_priority, reverse=True)
        if sorted_priority[0] != -1:
            selected_id = card_priority.index(sorted_priority[0])
            mask[selected_id] = 0
        
        # 检查是否有蓄爆牌，按优先级选择留下1-2张
        card_priority = [check_card_priority(card, self.stockpile_cards) for card in initial_hand]
        sorted_priority = sorted(card_priority, reverse=True)
        for i in range(2):
            if sorted_priority[i] != -1:
                selected_id = card_priority.index(sorted_priority[i])
                mask[selected_id] = 0

        # 如果充能+蓄爆牌共计3张，按优先级选择留下1张选留牌
        if sum(mask) == len(initial_hand) - 3:
            card_priority = [check_card_priority(card, self.optional_card) for card in initial_hand]
            sorted_priority = sorted(card_priority, reverse=True)
            if sorted_priority[0] != -1:
                selected_id = card_priority.index(sorted_priority[0])
                mask[selected_id] = 0

        return mask

