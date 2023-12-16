def string_to_ascii(input_string):
    ascii_list = [ord(char) for char in input_string]
    return ascii_list

# テスト用の文字列
input_string = input("文字列を入力してください: ")

# アスキーコードに変換
ascii_result = string_to_ascii(input_string)

# mooに変換
cow_result = ""
before_ascii = 0
for ascii_code in ascii_result:
    difference = ascii_code - before_ascii
    if(difference > 0):
        cow_result += " ".join(["MoO" for _ in range(difference)]) + " "
    if(difference < 0):
        cow_result += " ".join(["MOo" for _ in range(-difference)]) + " "
    before_ascii = ascii_code
    cow_result += "Moo\n"



# 結果の表示
print("Input String:", input_string)
print("ASCII Codes:", ascii_result)
print("cow Codes:", cow_result)