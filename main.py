
def main():
    train_loader, test_loader = set_loader(config)
    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    writer.add_graph(net, images)
    writer.close()

if __name__ == '__main__':
    main()