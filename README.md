
<body>

<h1>🚢 Titanic Survival Prediction</h1>
<h3>Professional Machine Learning Project | Logistic Regression</h3>

<div class="box">
<b>Author:</b> Musab Wali <br>
<b>Role:</b> Data Scientist / Machine Learning Engineer <br>
<b>Tools:</b> Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn
</div>

<hr>

<h2>📌 Project Objective</h2>
<p>
Is project ka goal yeh predict karna hai ke Titanic ke kaun se passengers
<b>survive</b> huay aur kaun se <b>nahi</b>.
Yeh ek <b>Binary Classification</b> problem hai.
</p>

<hr>

<h2>📂 Dataset Loading</h2>

<pre><code>import pandas as pd</code></pre>
<p>
<b>Explanation:</b><br>
<code>import</code> → Python keyword jo library load karta hai <br>
<code>pandas</code> → Data analysis library <br>
<code>as pd</code> → pandas ka short name
</p>

<pre><code>import numpy as np</code></pre>
<p>
<b>Explanation:</b><br>
<code>numpy</code> numerical calculations ke liye use hoti hai  
</p>

<pre><code>import matplotlib.pyplot as plt</code></pre>
<p>
<b>Explanation:</b><br>
Visualization ke liye matplotlib ka pyplot module
</p>

<pre><code>import seaborn as sns</code></pre>
<p>
<b>Explanation:</b><br>
Advanced & beautiful statistical graphs ke liye seaborn
</p>

<pre><code>df = pd.read_csv("titanic/test.csv")</code></pre>
<p>
<b>Explanation:</b><br>
<code>read_csv</code> → CSV file ko DataFrame mein load karta hai <br>
<code>df</code> → DataFrame ka variable
</p>

<hr>

<h2>🔍 Data Understanding</h2>

<pre><code>df.head()</code></pre>
<p>
Dataset ki pehli 5 rows show karta hai
</p>

<pre><code>df.shape</code></pre>
<p>
Rows aur columns ki total count batata hai
</p>

<pre><code>df.columns</code></pre>
<p>
Dataset ke tamam column names
</p>

<pre><code>df.info()</code></pre>
<p>
Data types aur missing values ki info
</p>

<pre><code>df.isnull().sum()</code></pre>
<p>
Har column mein kitni missing values hain
</p>

<hr>

<h2>🧹 Data Cleaning</h2>

<pre><code>
df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
</code></pre>

<p>
<b>Explanation:</b><br>
<code>drop</code> → Columns delete karta hai <br>
<code>axis=1</code> → Column-wise delete <br>
<code>inplace=True</code> → Original dataset modify hota hai
</p>

<pre><code>df['Age'] = df['Age'].fillna(df['Age'].median())</code></pre>
<p>
Missing Age values ko <b>median</b> se fill kiya
</p>

<pre><code>df['Fare'] = df['Fare'].fillna(df['Fare'].median())</code></pre>
<p>
Fare column ke missing values handle kiye
</p>

<hr>

<h2>🔄 Encoding Categorical Data</h2>

<pre><code>
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
</code></pre>
<p>
Text ko numeric mein convert kiya (ML ke liye zaroori)
</p>

<pre><code>
df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})
</code></pre>

<hr>

<h2>🎯 Feature & Target Split</h2>

<pre><code>
Y = df['Survived']
X = df.drop('Survived', axis=1)
</code></pre>

<p>
<b>X</b> → Features <br>
<b>Y</b> → Target variable
</p>

<hr>

<h2>✂️ Train Test Split</h2>

<pre><code>
from sklearn.model_selection import train_test_split
</code></pre>

<pre><code>
X_train, X_test, Y_train, Y_test =
train_test_split(X, Y, test_size=0.2, random_state=42)
</code></pre>

<p>
80% data training ke liye, 20% testing ke liye
</p>

<hr>

<h2>🤖 Model Training</h2>

<pre><code>
from sklearn.linear_model import LogisticRegression
</code></pre>

<pre><code>
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
</code></pre>

<p>
Logistic Regression binary classification ke liye use hota hai
</p>

<hr>

<h2>📊 Model Evaluation</h2>

<pre><code>
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, Y_pred)
</code></pre>

<p>
Model ki accuracy calculate karta hai
</p>

<pre><code>
confusion_matrix(Y_test, Y_pred)
</code></pre>

<p>
Correct & incorrect predictions ka summary
</p>

<hr>

<h2>🔁 Cross Validation</h2>

<pre><code>
cross_val_score(model, X, Y, cv=5)
</code></pre>

<p>
Model ki stability check karta hai
</p>

<hr>

<h2>🧪 New Passenger Prediction</h2>

<pre><code>
model.predict(new_passenger)
model.predict_proba(new_passenger)
</code></pre>

<p>
Real-world prediction aur survival probability
</p>

<hr>

<h2>⭐ Conclusion</h2>
<p>
Yeh project complete <b>end-to-end Data Science workflow</b> dikhata hai
jo industry-level expectations ko meet karta hai.
</p>

<h3>⭐ Star this repository if you like it!</h3>

</body>
</html>
