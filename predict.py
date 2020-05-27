import argparse
from utils import nn_model, json


parser = argparse.ArgumentParser(description='Load checkpoint e make a prediction')

parser.add_argument('--data-directory')

parser.add_argument('--checkpoint')

parser.add_argument('--top-k', type=int, default=5)

parser.add_argument('--category-names', default='./cat_to_name.json')

# parsing arguments
args = parser.parse_args()
print(args)
data_directory = args.data_directory
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names

# load classes categories
categories = json.load(category_names)

# load model

model, optimizer = nn_model.load(checkpoint)



predictions, probs = nn_model.predict(model=model, image_path=data_directory, topk=top_k)

result = list(zip(probs,predictions))
print('TOP {} PREDICTIONS:\n'.format(top_k))
for i in result:
    print('{}: {:.3}%'.format(categories[i[0]], i[1]*100))





