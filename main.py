from tests.DeepWalk_TSNE_test import DeepWalk_TSNE_test
from tests.SDNE_TSNE_test import SDNE_TSNE_test

if __name__ == "__main__":

    EMBED_METHODS = ("DeepWalk", "SDNE",)
    VIS_METHODS = ("TSNE",)

    while True:
        print("### Available:", end="  ")
        for name in EMBED_METHODS:
            print(name, end="  ")
        print()
        embed_method = input("(Case insensitive) Select embedding method: ")
        if embed_method.upper() in (name.upper() for name in EMBED_METHODS):
            break
        else:
            print("Invalid embedding method.")
    
    while True:
        print("### Available:", end="  ")
        for name in VIS_METHODS:
            print(name, end="  ")
        print()
        vis_method = input("(Case insensitive) Select visualizing method: ")
        if vis_method.upper() in (name.upper() for name in VIS_METHODS):
            break
        else:
            print("Invalid visualizing method.")
    
    print()

    embed_method = embed_method.upper()
    vis_method = vis_method.upper()

    if embed_method == "DEEPWALK":

        if vis_method == "TSNE":
            DeepWalk_TSNE_test()
    
    elif embed_method == "SDNE":

        if vis_method == "TSNE":
            SDNE_TSNE_test()