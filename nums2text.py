def to_text(n, pos, l, nums, suffix):
    if l <= 0:
        return ''
    if n[pos] == '0':
        return str(to_text(n, pos + 1, l - 1, nums, suffix))
    if n in nums:
        return nums[n]
    if l == 1:
        return nums[n[pos]]
    if l == 2:
        return str(nums[str(int(n[pos]) * 10)] + to_text(n, pos + 1, l - 1, nums, suffix))
    if l == 3:
        return str(nums[n[pos]]) + suffix[1] + to_text(n, pos + 1, l - 1, nums, suffix)
    text = ""
    length = l % 3
    if length == 0:
        length = 3
    suff = (l + 2) // 3 # 4,5,6->2;7,8,9->3;10,11,12->4, ...
    i = 0
    while i < l:
        text += to_text(n, i, length, nums, suffix)
        if int(n[i:i+length]) != 0 and l - i > 3:
            text += suffix[suff]
        suff -= 1
        i += length
        length = 3
    return text

nums = {'0': '', '1': "бір ", '2': "екі ", '3': "үш ", '4': "төрт ",
        '5': "бес ", '6': "алты ", '7': "жеті ", '8': "сегіз ", '9': "тоғыз ",
        "10": "он ", "20": "жиырма ", "30": "отыз ", "40": "қырық ", "50": "елу ",
        "60": "алпыс ", "70": "жетпіс ", "80": "сексен ", "90": "тоқсан "}
suffix = {0: '', 1: "жүз ", 2: "мың ", 3: "миллион ", 4: "миллиард ", 5: "триллион ",
          6: "квадриллион ", 7: "квинтиллион ", 8: "секстиллион ", 9: "септиллион ",
          10: "октиллион ", 11: "нониллион ", 12: "дециллион "}
for n in input().split(' '):
  if n == '0': print("нөл")
  else: print(to_text(n, 0, len(n), nums, suffix))
