from src.Model.auxiliary import *
from src.Model.hdbscan import HDBS

def main():
    sku_file = 'part-00000-tid-7173294866425216458-eae53fbf-d19e-4130-ba74-78f96b9675f1-4-1-c000.snappy.parquet'
    transactions_file = 'part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet'
    pdv_file = 'part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy.parquet'

    sku, transacoes, pdv = scan_parquets(sku_file, transactions_file, pdv_file)

    pdv_clusterizado = HDBS(pdv, transacoes, params={
            'min_cluster_size':100,
            'cluster_selection_epsilon':0.5,
            'cluster_selection_method':'leaf',
            'leaf_size':1000}
    ).hardcoded_cluster_pdv()

    reference_weeks = [52]
    benchmark = False
    prediction = train_model(period=8, transactions=transacoes, clusterized_pdv=pdv_clusterizado.lazy(), final_weeks=reference_weeks, benchmark=benchmark)
    

if __name__ == '__main__':
    main()




