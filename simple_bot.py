from gisim.classes.enums import ElementType

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
        self.is_first_round = True

    def choose_cards_to_replace(self, initial_hand):
        # initial_hand = active_player_info.hand_cards
        # 信息格式按照 active_player_info 来
        # 开局初始化
        mask = [1] * 5  # 初始化mask，表示所有牌都可以替换

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

        # 返回选择要替换的牌
        return mask

    def reroll_dice(self, active_player_info):
        dices = [str(dice) for dice in active_player_info.dice_zone]
        cur_chara_pos = active_player_info.active_character_position.value
        if active_player_info.characters[0].character.alive:
            # 可莉存活（一定在[0]位置），保留当前角色与火元素/万能骰子
            cur_chara_element_type = active_player_info.characters[cur_chara_pos].character.element_type
            dice_to_keep = [ElementType.PYRO, ElementType.OMNI, cur_chara_element_type, ]
        else:
            dice_to_keep = [ElementType.OMNI,]
            if active_player_info.characters[1].character.alive:
                # xingqiu存活
                dice_to_keep.append(ElementType.HYDRO)
            if active_player_info.characters[2].character.alive:
                # Kaeya存活
                dice_to_keep.append(ElementType.CRYO)
            
        mask = [0 if dice in dice_to_keep else 1 for dice in dices] # mask中为1表示需要重掷
        return mask

    def switch_active_role(self, active_player_info):
        if self.is_first_round:
            choosen_role = 1    # 选择行秋
        else:
            if active_player_info.characters[0].character.alive:
                choosen_role = 0    # 可莉存活，选择可莉
            else:
                choosen_role = 2    # 可莉阵亡，选择凯亚
        return choosen_role # 考虑返回序号还是类似mask
    
    def check_latent_dice(self, active_player_info):
        pass
    
    def choose_stockpile_card(self, active_player_info):
        cur_chara_pos = active_player_info.active_character_position.value
        cur_chara_element_type = active_player_info.characters[cur_chara_pos].character.element_type
        hand_cards = active_player_info.hand_cards
        dices = [str(dice) for dice in active_player_info.dice_zone]
        unimportant_cards_num = len(hand_cards)
        for important_card in self.stockpile_cards + self.optional_card:
            unimportant_cards_num -= hand_cards.count(important_card)
        # 实际可使用的骰子数 = 非关键手牌+出战角色元素骰子+万能骰子
        available_dices_num = unimportant_cards_num + dices.count(cur_chara_element_type) + \
            dices.count(ElementType.OMNI)
        for stockpile_card in self.stockpile_cards:
            if stockpile_card in hand_cards and stockpile_card not in active_player_info.support_zone:
                # 目前手牌有此卡牌且支援区无此卡牌
                # 检查骰子是否足够此卡牌使用，即出战角色元素骰子+万能骰子+非关键手牌>=3（暂定）
                # if available_dices_num - stockpile_card.cost >=0
                #   return hand_cards.index(stockpile_card)
                pass


