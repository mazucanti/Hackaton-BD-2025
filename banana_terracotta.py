from src.Model.auxiliary import *
from src.Model.hdbscan import HDBS
from src.Model.thompson_sampling import ThompsonSampling
from pathlib import Path

def main():
    sku_file = Path('data/part-00000-tid-7173294866425216458-eae53fbf-d19e-4130-ba74-78f96b9675f1-4-1-c000.snappy.parquet')
    transactions_file = Path('data/part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet')
    pdv_file = Path('data/part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy.parquet')

    sku, transacoes, pdv = scan_parquets(sku_file, transactions_file, pdv_file)

    pdv_clusterizado = HDBS(pdv, transacoes, params={
            'min_cluster_size':100,
            'cluster_selection_epsilon':0.5,
            'cluster_selection_method':'leaf',
            'leaf_size':1000}
    ).hardcoded_cluster_pdv()

    ts = ThompsonSampling(transactions_file, pdv_file)
    ts.get_probabilities()

    prob = ts.scale_probs()


    reference_weeks = [52]
    benchmark = False
    for k, lim in enumerate([0.8, 0.9, 0.986, 0.986, 0.99]):
        prediction = train_model(
            period=8,
            transactions=transacoes,
            clusterized_pdv=pdv_clusterizado.lazy(),
            final_weeks=reference_weeks,
            benchmark=benchmark,
            threshold=lim
        )
    
    result = prob.lazy().cast(
        {
            'pdv': pl.Categorical
        }
    ).join(
        prediction.lazy(), how='right', left_on='pdv', right_on='internal_store_id'
    ).with_columns(
        column_0=pl.col('scaled_prob')*pl.col('column_0'),
        column_1=pl.col('scaled_prob')*pl.col('column_1'),
        column_2=pl.col('scaled_prob')*pl.col('column_2'),
        column_3=pl.col('scaled_prob')*pl.col('column_3')
    ).drop('scaled_prob').explode(['column_0', 'column_1', 'column_2', 'column_3', 'internal_product_id'])

    print(result.unpivot(cs.starts_with('column')).collect())



if __name__ == '__main__':
    main()




