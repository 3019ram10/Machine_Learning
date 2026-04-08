import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import networkx as nx
print("MARKET BASKET ANALYSIS")

transactions = [
    ['Bread', 'Milk', 'Eggs'],
    ['Bread', 'Butter', 'Jam'],
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Milk', 'Butter', 'Eggs'],
    ['Milk', 'Eggs', 'Cereal'],
    ['Bread', 'Eggs', 'Milk'],
    ['Butter', 'Jam', 'Bread'],
    ['Bread', 'Milk', 'Butter'],
    ['Milk', 'Cereal', 'Eggs'],
    ['Bread', 'Butter', 'Jam', 'Milk']
]

print(f"\nTotal transactions: {len(transactions)}")
print("\nSample transactions:")
for i in range(3):
    print(f"Transaction {i+1}: {transactions[i]}")
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

print("\nBinary matrix (first 5 rows):")
print(df.head())

print("BASIC STATISTICS")

item_freq = df.sum().sort_values(ascending=False)
print("\nItem frequencies:")
print(item_freq)
trans_lengths = [len(t) for t in transactions]
print(f"\nAverage items per transaction: {np.mean(trans_lengths):.2f}")
print(f"Min items: {min(trans_lengths)}")
print(f"Max items: {max(trans_lengths)}")

print("APRIORI RESULTS")


min_support = 0.3
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
print(f"\nFrequent itemsets (support >= {min_support}):")
print(f"Total: {len(frequent_itemsets)}")
print(frequent_itemsets)


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

print(f"\nAssociation rules generated: {len(rules)}")
print("\nAll rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])


rules = rules[rules['confidence'] >= 0.5]
print(f"\nRules with confidence >= 0.5: {len(rules)}")

print("TOP RULES")

print("\nTOP 5 RULES BY LIFT:")
top_lift = rules.nlargest(5, 'lift')
for i, (idx, row) in enumerate(top_lift.iterrows(), 1):
    ante = ', '.join(list(row['antecedents']))
    cons = ', '.join(list(row['consequents']))
    print(f"{i}. {ante} → {cons}")
    print(f"   Support: {row['support']:.2f}, Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f}")
print("\nTOP 5 RULES BY CONFIDENCE:")
top_conf = rules.nlargest(5, 'confidence')
for i, (idx, row) in enumerate(top_conf.iterrows(), 1):
    ante = ', '.join(list(row['antecedents']))
    cons = ', '.join(list(row['consequents']))
    print(f"{i}. {ante} → {cons}")
    print(f"   Support: {row['support']:.2f}, Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f}")

plt.figure(figsize=(10, 6))
plt.bar(range(len(item_freq)), item_freq.values)
plt.xticks(range(len(item_freq)), item_freq.index, rotation=45)
plt.title('Item Frequencies')
plt.xlabel('Items')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

if len(rules) > 0:
    plt.figure(figsize=(8, 6))
    G = nx.DiGraph()

    for idx, row in top_lift.iterrows():
        ante = ', '.join(list(row['antecedents']))
        cons = ', '.join(list(row['consequents']))
        G.add_edge(ante, cons, weight=row['lift'])
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=2000, font_size=10, font_weight='bold',
            arrows=True, arrowstyle='->', arrowsize=20)
    plt.title('Top Rules Network')
    plt.show()
    print("BUSINESS INTERPRETATION")
print("\nWhat the rules mean:")
print("• If a customer buys [item A], they are likely to also buy [item B]")
print("• Lift > 1 means items are positively associated")
print("• Confidence shows how often the rule holds true")
print("\nKey Findings:")
if len(rules) > 0:
    best_rule = rules.nlargest(1, 'lift').iloc[0]
    ante = ', '.join(list(best_rule['antecedents']))
    cons = ', '.join(list(best_rule['consequents']))
    print(f"• Strongest association: {ante} → {cons}")
    print(f"  (Lift: {best_rule['lift']:.2f})")
print("\nMarketing Suggestions:")
print("1. Store Layout: Place associated items near each other")
print("2. Promotions: Offer discounts on pairs that are frequently bought together")
print("3. Cross-selling: Train staff to suggest complementary items")
print("4. Product placement: Create themed displays (e.g., breakfast section)")

# Save results
rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_csv('simple_rules.csv', index=False)
print("\nResults saved to 'simple_rules.csv'")