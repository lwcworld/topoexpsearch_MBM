from Utils_graph2vec import *
import glob
from tqdm import tqdm
from joblib import Parallel, delayed

if __name__ == "__main__":
    """
    Main function to read the graph list, extract features.
    Learn the embedding and save it.
    :param args: Object with the arguments.
    """
    args = {'input_path': '../dataset/dataset_PO_floorplan_c5/',
            'output_path': '../dataset/feature_PO_floorplan_c5/dim8.csv',
            'dimensions': 8,
            'workers': 8,
            'epochs': 100,
            'min_count': 0,
            'wl_iterations': 2,
            'learning_rate': 0.025,
            'down_sampling': 0.0001}

    graphs = glob.glob(args['input_path'] + "*.json")
    print("\nFeature extraction started.\n")
    document_collections = Parallel(n_jobs=args['workers'])(
        delayed(feature_extractor)(g, args['wl_iterations']) for g in tqdm(graphs))
    print("\nOptimization started.\n")

    model = Doc2Vec(document_collections,
                    vector_size=args['dimensions'],
                    window=0,
                    min_count=args['min_count'],
                    dm=0,
                    sample=args['down_sampling'],
                    workers=args['workers'],
                    epochs=args['epochs'],
                    alpha=args['learning_rate'])

    model.save('../dataset/model/graph_embedding/model_embedding'+'_dim'+str(args['dimensions']))
    save_embedding(args['output_path'], model, graphs, args['dimensions'])
    print('finished!!')

