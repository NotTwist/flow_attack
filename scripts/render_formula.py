import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

formula = r"$\theta_{t+1} = \theta_t - \alpha \cdot \mathrm{sign}\!\left(\nabla_{\!\theta}\,\mathcal{L}(\theta_t)\right)$"

fig, ax = plt.subplots(figsize=(9, 1.6))
ax.set_axis_off()
fig.patch.set_facecolor('white')

ax.text(0.5, 0.5, formula, transform=ax.transAxes,
        fontsize=38, ha='center', va='center',
        fontfamily='serif')

out = 'Diploma__review_/figures/fgsm_step.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=220, bbox_inches='tight', facecolor='none', transparent=True)
print(f'Saved -> {out}')
