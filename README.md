# Supplier Performance & Efficiency Analysis (MCDA · DEA · MOLP)

**Evaluating 12 suppliers of a crystal retailer on quality, service and efficiency, and setting improvement targets for the inefficient ones**

MSc Business Analytics group project, University of Manchester

`Python (SciPy)` · `Excel Solver` · `Evidential Reasoning` · `AHP` · `Data Envelopment Analysis` · `Multi-Objective LP`

---

## 📌 Business question
A retailer of hand-made crystal products wants to expand regionally and depends on just-in-time deliveries during festival seasons. Management needs to know:
1. Which suppliers deliver the best **product quality** and **customer service**?
2. Which suppliers are **efficient**, given what they cost and how reliably they deliver?
3. What **realistic improvement targets** should the inefficient suppliers be set?

## 🔧 Method
**1. Multi-criteria assessment (MCDA): Evidential Reasoning**
- 11 criteria in two groups. Product quality: durability, defects, innovation, variety, packaging, brand. Service: after-sales, warranty, communication, stock, minimum order.
- Weights were derived with **AHP** (consistency ratio < 0.10). Utilities were elicited from 8 decision makers and show a risk-averse (concave) shape.
- Each supplier receives a single utility score for quality and one for service. A sensitivity check with swing weighting left the rankings almost unchanged.

**2. Efficiency analysis: input-oriented BCC DEA** (`excel/`, `src/dea_bcc.py`)
- **Inputs:** unit price, late delivery %, shipping error %. Lead time was dropped because it correlates with late delivery (r = 0.52).
- **Outputs:** the product-quality and customer-service utilities from step 1.
- Solved for each supplier as a linear programme, with one Excel Solver sheet per supplier, and **re-implemented in Python with SciPy**, which reproduces the Solver results exactly.

**3. Improvement targets: MOLP (DEA-IMRP)** (`src/dea_imrp.py`)
- Standard DEA asks for the same proportional cut in every input. The BCC model is reframed as a **multi-objective LP**, and the **interactive minimax reference point method with gradient projection** searches the whole efficient frontier for targets that match management priorities: late delivery first, then shipping errors, then price.
- Pipeline for each inefficient supplier:
  1. **Payoff table** gives the ideal point f\* (each input minimised on its own).
  2. **Minimax LP** with weights wᵢ = 1/(xᵢ − fᵢ\*) gives the first efficient point.
  3. **Gradient projection**: the frontier normal comes from the LP duals (N = w·β). The decision maker's preference direction is projected onto the frontier, a step γ·α_max is taken, the weights are updated and the LP is solved again.
  4. The loop stops when the move size ‖α·Δ‖ falls below the noise threshold.
- Fully implemented in Python. It reproduces every iteration reported in the analysis to 4 decimal places.

## 📈 Key results

| Supplier | Efficiency θ* | Main benchmark |
|---|---|---|
| A, B, D, G, H, I, K | 1.000 (efficient) | – |
| J | 0.926 | I (λ = 0.82) |
| L | 0.703 | K, B |
| C | 0.656 | I, B |
| F | 0.522 | I |
| **E** | **0.429** | I (λ = 0.91) |

- **Supplier H** scores highest on both quality (u = 0.80) and service (u = 0.69) and is efficient, so it is the **recommended primary supplier**. **A** and **B** are strong alternatives.
- **Supplier E** is the least efficient: it would need to cut its inputs by about 57% to reach the frontier.
- The MOLP targets accept a slightly **higher unit price in exchange for far fewer late deliveries**, which suits a premium, made-to-order product:

| Supplier | Unit price | Late delivery | Shipping error |
|---|---|---|---|
| C | $2.13 → $1.81 | 6.0% → 1.97% | 6.0% → 3.51% |
| F | $2.98 → $1.65 | 4.0% → 1.34% | 6.0% → 3.28% |
| L | $1.94 → $1.62 | 9.0% → 2.11% | 6.0% → 3.30% |
| E | $3.44 → $1.40 | 4.0% → 1.10% | 7.0% → 3.12% |
| J | $1.50 → $1.40 | 6.0% → 1.14% | 4.0% → 3.15% |
- **Recommendation:** a tiered supplier strategy, with core volume to H, A and B, backup suppliers for resilience, and measurable delivery targets for C, E, F, J and L.

## 📁 Repository structure
```
├── supplier_dea_imrp.ipynb  # full analysis with outputs (BCC DEA + DEA-IMRP)
├── excel/
│   └── DEA_Dual_BCC.xlsx    # data, one Solver LP sheet per supplier, results summary
├── src/
│   ├── dea_bcc.py           # BCC DEA efficiency scores (SciPy)
│   └── dea_imrp.py          # DEA-IMRP: payoff table, minimax LP, gradient projection
└── requirements.txt
```

## ▶️ How to run
```bash
pip install -r requirements.txt
jupyter notebook supplier_dea_imrp.ipynb
# or run the scripts: cd src && python dea_bcc.py && python dea_imrp.py
```
The Excel model needs the **Solver add-in** (yellow cells = decision variables, orange cell = objective).

---
**Author:** Sophia Su · [LinkedIn](https://www.linkedin.com/in/yi-wen-su-523a54433)
