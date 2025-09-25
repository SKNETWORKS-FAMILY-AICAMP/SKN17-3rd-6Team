import os

os.chdir('./results')
filenames = os.listdir()

print(filenames)

for filename in filenames:
    print(os.path.splitext(filename)[0].split('_')[-1])