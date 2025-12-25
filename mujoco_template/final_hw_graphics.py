import csv
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

results = []
with open("tradeoff_results.csv","r") as f:
    for r in csv.DictReader(f):
        r["RMSE"] = float(r["RMSE"])
        r["Chattering"] = float(r["Chattering"])
        if r["phi"] != "N/A":
            r["phi"] = float(r["phi"])
        results.append(r)


plt.figure(figsize=(8,6))
for r in results:
    if r["Controller"] == "ID":
        plt.scatter(r["Chattering"], r["RMSE"], marker="o", label="ID", s=100)
    else:
        plt.scatter(r["Chattering"], r["RMSE"], marker="x", label=f"SMC φ={r['phi']}", s=100)

plt.xlabel("Chattering  (‖s‖ mean)")
plt.ylabel("RMSE (tracking error)")
plt.title("Robustness vs Chattering Trade-Off")
plt.grid(True)
plt.legend()

plt.savefig("tradeoff_plot.png", dpi=300)
print("Saved: tradeoff_plot.png")