import matplotlib.pyplot as plt

def make_plot(img, lbl, pred):
    plt.figure(figsize=(10,10))
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title('Input image')
    plt.subplot(1,3,2)
    plt.imshow(lbl, vmin=0, vmax=1, cmap='gray')
    plt.title('GT mask')
    plt.subplot(1,3,3)
    plt.imshow(pred, vmin=0, vmax=1, cmap='gray')
    plt.title('Model output')
    plt.show()
