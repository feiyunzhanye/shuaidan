# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 09:36:11 2019

@author: ChinaNet
"""

import jieba
jieba.load_userdict("data/zyx_dict.txt")
import jieba.analyse
content = "201802-十全十美99元套餐2018版（专营）(2-2A8EM312)|| || ||移动主卡：新装 后付费CDMA ||UIM卡号：********||选号备注：主卡号码随机 ||201801-锦江合作送2500积分促销(2-29O4VMPU) || 201412全家享新融合促销活动，协议期12个月(2-108CS3FI) || || || 移动副卡:新装 || 张数:1 张 || UIM卡号: ********|| 选号备注: 副卡号码随机|| ||宽带: 已有 ||宽带设备号: ********||备注：如果原宽带设备下的存在IPTV和爱BABY，则全部转入 ||宽带计费速率: 包月制（100M/4M） ||接入方式: FTTH ||201712-全家享新融合99档次宽带加装包（标签网龄）(2-27J39G74) || || ||||备注：新装场景新建分账序号，存量场景使用存量设备分账序号，移动新装号码必须开通翼支付功能和4G功能||||联系方式：********"
jieba.analyse.set_stop_words("data/stop_words.txt")
seg_list = jieba.analyse.extract_tags(content,topK=30, withWeight=False, allowPOS=('n','v'))
for seg in seg_list:
    print(seg)