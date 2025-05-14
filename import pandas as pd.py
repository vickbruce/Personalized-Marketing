import pandas as pd
from sklearn.cluster import KMeans

data = {
    'customer_id': ['C101', 'C102', 'C103', 'C104', 'C105'],
    'name': ['Riya', 'Karan', 'Meena', 'Ayaan', 'Sneha'],
    'email': ['riya@domain.com', 'karan@domain.com', 'meena@domain.com', 'ayaan@domain.com', 'sneha@domain.com'],
    'total_spent': [1450, 1200, 350, 2600, 900],
    'purchase_frequency': [4, 12, 3, 2, 10]
}

df = pd.DataFrame(data)

X = df[['total_spent', 'purchase_frequency']]
kmeans = KMeans(n_clusters=3, random_state=42)
df['segment'] = kmeans.fit_predict(X)

segment_offers = {
    0: "10% off on your next purchase!",
    1: "Exclusive access to premium deals!",
    2: "Free shipping + surprise gift!"
}

def generate_segment_email(name, segment):
    offer = segment_offers[segment]
    return f"""
Hi {name},

Thanks for being a valued customer! Based on your recent activity, we’ve got something special for you:

{offer}

Visit our store and make the most of your benefits!

Cheers,  
AI Marketing Bot
"""

print("********** Personalized Email Campaign **********")
for _, row in df.iterrows():
    email = generate_segment_email(row["name"], row["segment"])
    print(email)
    print("*" * 50)
