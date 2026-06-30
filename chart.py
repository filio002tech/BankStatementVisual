from openpyxl import Workbook
import matplotlib.pyplot as plt

wb=Workbook()
ws=wb.active
ws.title="System Evaluation"
ws.append(["Metric","Score (%)"])
data=[
("Login Authentication",100),
("User Registration",100),
("File Upload",95),
("File Encryption",100),
("File Sharing",95),
("File Decryption",100),
("Database Response",90),
("Overall Performance",97),
]
for r in data: ws.append(r)
xlsx="/mnt/data/System_Evaluation_Table.xlsx"
wb.save(xlsx)

metrics=[d[0] for d in data]
scores=[d[1] for d in data]
plt.figure(figsize=(8,4.5))
plt.bar(range(len(metrics)),scores)
plt.xticks(range(len(metrics)),metrics,rotation=35,ha='right')
plt.ylim(0,110)
plt.ylabel("Score (%)")
plt.title("System Performance Evaluation")
plt.tight_layout()
chart="/mnt/data/System_Performance_Evaluation.png"
plt.savefig(chart)
plt.close()

print({"xlsx":xlsx,"chart":chart})
