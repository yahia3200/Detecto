import pickle
import os
import sys
from DataLoading import ImagesLoader
import time

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Wrong Number of Arguments!")
        sys.exit(-1)

    imgs_paths = os.listdir(sys.argv[1])
    imgs_paths = sorted(imgs_paths, key=lambda x: int(x[:-4]))

    y_pred = []
    times = []

    # Load Model
    data_pipe = pickle.load(open('model/data_pipeline', 'rb'))
    model_pip = pickle.load(open('model/model_pipeline', 'rb'))

    # Load Data
    for img_path in imgs_paths:
        image = ImagesLoader([f"{sys.argv[1]}{img_path}"])
        start = time.time()
        test_point = data_pipe.transform(image)
        pred = model_pip.predict(test_point)
        end = time.time()
        y_pred.append(pred)
        times.append(end - start)

    # Save Results
    with open(f"{sys.argv[2]}results.txt", "w") as out:
        for pred in y_pred:
            out.write(str(pred[0]))
            out.write('\n')
    # Save Times
    with open(f"{sys.argv[2]}times.txt", "w") as out:
        for t in times:
            out.write(str(max(0.01, round(t, 2))))
            out.write('\n')
