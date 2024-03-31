import itertools
import pandas as pd
import numpy as np

def generate_weight_vector(k):
    """Generate weight vectors based on the specified combination."""
    weight_vector = []
    values = [round(x * 0.1, 1) for x in range(0, 11)]
    combinations = list(itertools.product(values, repeat=k))
    for comb in combinations:
        weight_vector.append(list(comb))
    return weight_vector

def load_pd_genes(filename):
    """Load PD genes from the given file."""
    pd_genes_reported = pd.read_csv(filename, header=None)
    return pd_genes_reported[0].tolist()

def calculate_scores(score_matrix, pd_genes_reported,K,PC,PD_Ass):
    """Calculate scores for the result DataFrame."""
    combination=pow(11,K)
    A=score_matrix.iloc[:-2, 1:combination+1].mul(score_matrix['PD_Assoc'].iloc[:-2], axis=0).sum()
    B=score_matrix.iloc[:-2, 1:combination+1].sum()
    wm = pd.DataFrame(np.array(generate_weight_vector(K)))
    wm['match_genes'] = A.to_list()
    wm['sel_genes'] =B.tolist()
    wm['score'] = (A/B).to_list()
    wm['TP'] = wm['match_genes']
    wm['FN'] = PD_Ass - wm['match_genes']
    wm['FP'] = wm['sel_genes']
    wm['TN'] = (PC - PD_Ass) - wm['sel_genes']
    wm['F1_Score'] = wm.apply(calculate_f1_score, axis=1)
    return wm
	
def calculate_f1_score(row):
    """Calculate F1 score."""
    precision = row['TP'] / (row['TP'] + row['FP'])
    recall = row['TP'] / (row['TP'] + row['FN'])
    if precision + recall == 0:
        return 0  # Handle the case where precision + recall is zero to avoid division by zero
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def calculate_decision_vector(Know, K,pd_genes_reported,weights_matrix, cutoff):
    """Calculate decision vector based on input data."""
    combination = pow(11, K)
    data_to_multiply = Know.iloc[:, -K:]
    decision_matrix = np.dot(data_to_multiply, weights_matrix)
    result_column_names = [f'DV_{i + 1}' for i in range(decision_matrix.shape[1])]
    decision_vector = pd.DataFrame(decision_matrix, columns=result_column_names)
    decision_vector = (decision_vector.iloc[:, -combination:] > cutoff).astype(int)
    decision_vector = pd.concat([Know['SYMBOL'], decision_vector], axis=1)
    decision_vector['PD_Assoc'] = decision_vector['SYMBOL'].apply(lambda x: 1 if x in pd_genes_reported else 0)
    df = decision_vector.iloc[:, 1:combination+1].mul(decision_vector['PD_Assoc'], axis=0)
    df1 = decision_vector.iloc[:, 1:combination+1].sum()
    new_row = df.sum()
    decision_vector = decision_vector.append(df1, ignore_index=True)
    decision_vector = decision_vector.append(new_row, ignore_index=True)
    return decision_vector
	
# Read data
KM = pd.read_csv('Knolwedge_matrix.csv')
sel_columns=['SYMBOL','Adult_b','Embryo_b','ND_b']
M = KM[sel_columns]
K = 3 # TOtal Numbe rof Knowlegde Used
cutoff = 0.5
PC = 17374 #TOtal protein Coding genes of the given dataset
PD_Ass = 272 # #Total genes reported to be associated with PD by GWAS Catalogue. 
weight_vectors = generate_weight_vector(K)
weights_matrix = np.array(weight_vectors).T
Know = M
pd_genes_reported = load_pd_genes('PD_genes_reported.csv')
decision  = calculate_decision_vector(Know,K,pd_genes_reported,weights_matrix,cutoff)
wm = calculate_scores(decision , pd_genes_reported,K,PC,PD_Ass)
new_column_names = {0: 'Adult', 1: 'Embryo', 2: 'ND'}
wm = wm.rename(columns=new_column_names)
product = wm['match_genes'] * wm['F1_Score']
max_product_index = product.idxmax() 
row_with_max_product = wm.loc[max_product_index]
selgenes_df = decision .loc[:, decision .iloc[-1]==row_with_max_product['match_genes']]
Hybsel_genes=decision [selgenes_df.iloc[:,0]==1]['SYMBOL']
Hybsel_genes.to_csv('HybSel_genes.txt', header=False, index=False)

#Hybsel genes SNPS Mapping 
grch37 = pd.read_csv('Genes_Grch37.csv')
genotype = pd.read_table('MyData11.bim', delim_whitespace=True)
genotype.columns =['CHR', 'SNP','DKN','POS','REF','ALT']
genotype['CHR'] = genotype['CHR'].astype(str)
grch37['CHR'] = grch37['CHR'].astype(str)
merged_df = pd.merge(genotype, grch37, on='CHR')
result_df = merged_df[(merged_df['POS'] >= merged_df['START']) & (merged_df['POS'] <= merged_df['END'])]
SNP_Gene_mapper=result_df[['CHR','SNP','POS','SYMBOL']].copy()
HybSel_snps = SNP_Gene_mapper[SNP_Gene_mapper['SYMBOL'].isin(Hybsel_genes.iloc[:, 0])]
HybSel_snps.to_csv('HybSel_SNPs.txt', header=False, index=False)
