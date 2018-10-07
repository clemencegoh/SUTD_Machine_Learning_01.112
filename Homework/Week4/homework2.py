from Homework.Week2.dataset import CSV_DATA


dataset = "D:\desktopStorage\school\SUTD_Machine_Learning_01.112\Homework\Week2\www.dropbox.com\s\oqoyy9p849ewzt2\linear.csv"


def ridgeRegressionWithOffset():
    with open(dataset) as f:
        for line in f:
            print(line)



if __name__ == '__main__':
    ridgeRegressionWithOffset()
