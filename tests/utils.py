def print_block(title):
    print()
    if (len(title) > 52):
        length = len(title)
        print("### " + "=" * length + " ###")
        print("### " + " " * length + " ###")
        print("### " + title + " ###")
        print("### " + " " * length + " ###")
        print("### " + "=" * length + " ###")
    else:
        print("### " + "========== " * 5 + "###")
        print("###" + " " * 56 + "###")
        left = (56 - len(title)) // 2
        right = 56 - len(title) - left
        print("###" + " " * left + title + " " * right + "###")
        print("###" + " " * 56 + "###")
        print("### " + "========== " * 5 + "###")
    print()

def get_index():
    while True:
        try:
            i = int(input("Enter test index: "))
            break
        except:
            print("Invalid index.")
    return i