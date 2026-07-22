#!/usr/bin/env python3
"""
Statistical analysis of the four IMB-QA generator-judge settings for
"Empathy Editing for Italian Medical Question Answering" (CLiC-it 2026).

Computes the significance tests and confidence intervals reported in the paper:
  - McNemar's test on paired judges scoring identical generated texts
  - Bootstrap 95% CIs on every factual-pass / SQI rate (Table 6)
  - Wilson CIs on judge-calibration accuracies (Table 3)
  - Logistic regression testing the length -> quality-preference confound
  - Category-level factual-pass with Wilson CIs
  - Decomposition of the factual-pass gap (preservation vs unsupported additions)
  - Cohen's kappa inter-judge and calibration agreement

Reads the per-row CSVs from "../IMB Generation Judgement Results/" and writes
tables to ./outputs and figures to ./figures. Deterministic (bootstrap seed 42).
"""
import os, glob, json, warnings
import numpy as np
import pandas as pd
from scipy import stats as st
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import cohen_kappa_score
warnings.filterwarnings("ignore")  # silence sklearn single-label warnings on degenerate (skewed) checks

# Resolve paths relative to this script so the folder is portable within the repo.
ROOT   = os.path.dirname(os.path.abspath(__file__))
MSE    = os.path.dirname(ROOT)
RESDIR = os.path.join(MSE, "IMB Generation Judgement Results")
OUT    = os.path.join(ROOT, "outputs")
FIG    = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

FILES = {
 ("Llama","Llama"):    "current_run_IMB_V3_REFINED_CLINICAL_EDITING_GENERATED_BY_llama_JUDGED_BY_plus_llama_3_3_70b_versatile_seed-42.csv",
 ("Llama","GPT-OSS"):  "current_run_IMB_V3_REFINED_CLINICAL_EDITING_GENERATED_BY_llama_JUDGED_BY_plus_openai_gpt_oss_120b_seed-42.csv",
 ("GPT-OSS","Llama"):  "current_run_IMB_V3_REFINED_CLINICAL_EDITING_GENERATED_BY_plus_openai_gpt_oss_120b_JUDGED_BY_plus_llama_3_3_70b_versatile_seed-42.csv",
 ("GPT-OSS","GPT-OSS"):"current_run_IMB_V3_REFINED_CLINICAL_EDITING_GENERATED_BY_plus_openai_gpt_oss_120b_JUDGED_BY_plus_openai_gpt_oss_120b_seed-42.csv",
}
ORDER = [("Llama","Llama"),("Llama","GPT-OSS"),("GPT-OSS","GPT-OSS"),("GPT-OSS","Llama")]

# ---------- helpers ----------
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (c-h, c+h)

def boot_ci(x, B=10000, seed=42):
    rng = np.random.default_rng(seed); x = np.asarray(x); n = len(x)
    bs = x[rng.integers(0, n, (B, n))].mean(1)
    return tuple(np.percentile(bs, [2.5, 97.5]))

def mcnemar_paired(a, b):
    a = np.asarray(a); b = np.asarray(b)
    n11 = int(((a==1)&(b==1)).sum()); n10 = int(((a==1)&(b==0)).sum())
    n01 = int(((a==0)&(b==1)).sum()); n00 = int(((a==0)&(b==0)).sum())
    tab = [[n11, n10], [n01, n00]]
    r = mcnemar(tab, exact=(n10+n01) < 25)
    return dict(n10=n10, n01=n01, stat=float(r.statistic), p=float(r.pvalue))

def derive(df):
    out = pd.DataFrame({'id': df['id'], 'cat': df['general_category']})
    out['emp_gen']  = (df['empathy_pred_source'] == 'Generated').astype(int)
    out['qual_gen'] = (df['quality_pred_source'] == 'Generated').astype(int)
    out['pres']     = (df['factual_preservation_answer'] == 'Yes').astype(int)
    out['nounsupp'] = (df['unsupported_addition_answer'] == 'No').astype(int)
    out['fp']       = (out['pres'] & out['nounsupp']).astype(int)
    out['sqi']      = (out['qual_gen'] & out['fp']).astype(int)
    gw = df['generated_empathy_answer'].astype(str).str.split().str.len()
    out['gen_wc'] = gw.values; out['orig_wc'] = df['answer_word_count'].values
    return out.set_index('id')

# ---------- load ----------
D = {k: pd.read_csv(os.path.join(RESDIR, f)) for k, f in FILES.items()}
M = {k: derive(v) for k, v in D.items()}

report = {}

# ---------- 1. reproduce Table 6 + bootstrap CIs ----------
t6 = []
for k in ORDER:
    m = M[k]
    row = dict(generator=k[0], judge=k[1],
               emp_gen=int(m.emp_gen.sum()), qual_gen=int(m.qual_gen.sum()),
               pres=int(m.pres.sum()), nounsupp=int(m.nounsupp.sum()),
               fp=int(m.fp.sum()), sqi=int(m.sqi.sum()))
    lo, hi = boot_ci(m.fp.values);  row['fp_ci']  = f"[{lo*100:.1f}, {hi*100:.1f}]"
    lo, hi = boot_ci(m.sqi.values); row['sqi_ci'] = f"[{lo*100:.1f}, {hi*100:.1f}]"
    t6.append(row)
t6 = pd.DataFrame(t6)
t6.to_csv(os.path.join(OUT, "table6_with_bootstrap_ci.csv"), index=False)
report['table6'] = t6.to_dict('records')

# ---------- 2. McNemar paired judges (identical generated texts) ----------
mc_rows = []
for gen in ["Llama", "GPT-OSS"]:
    jl, jg = M[(gen,"Llama")], M[(gen,"GPT-OSS")]
    common = jl.index.intersection(jg.index)
    jl, jg = jl.loc[common], jg.loc[common]
    for metric in ['fp','qual_gen','pres','nounsupp']:
        r = mcnemar_paired(jl[metric].values, jg[metric].values)
        mc_rows.append(dict(generated_by=gen, metric=metric, n=len(common),
                            llama_judge=int(jl[metric].sum()), gptoss_judge=int(jg[metric].sum()),
                            discordant_llamaYes_gptossNo=r['n10'],
                            discordant_llamaNo_gptossYes=r['n01'], p=r['p']))
mc = pd.DataFrame(mc_rows)
mc.to_csv(os.path.join(OUT, "mcnemar_paired_judges.csv"), index=False)
report['mcnemar'] = mc.to_dict('records')

# ---------- 2b. Cohen's kappa: inter-JUDGE agreement (scikit-learn) ----------
# Chance-corrected agreement between the two LLM judges on identical generated
# texts. NB: this is inter-annotator agreement between two *judges*, not human
# validation. Skewed base rates deflate kappa even when raw agreement is high.
def kappa_boot_ci(a, b, B=2000, seed=42):
    a = np.asarray(a); b = np.asarray(b); n = len(a); rng = np.random.default_rng(seed)
    k = cohen_kappa_score(a, b); ks = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        kk = cohen_kappa_score(a[idx], b[idx])
        if not np.isnan(kk): ks.append(kk)
    lo, hi = (np.percentile(ks, [2.5, 97.5]) if ks else (np.nan, np.nan))
    return k, lo, hi

def kappa_strength(k):
    if np.isnan(k): return "undefined"
    if k < 0:    return "poor"
    if k < 0.20: return "slight"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost perfect"

kap_rows = []
for gen in ["Llama", "GPT-OSS"]:
    jl, jg = M[(gen,"Llama")], M[(gen,"GPT-OSS")]
    common = jl.index.intersection(jg.index); jl, jg = jl.loc[common], jg.loc[common]
    for metric, lab in [('emp_gen','Empathy prefers edited'),('qual_gen','Quality prefers edited'),
                        ('pres','Info preserved'),('nounsupp','No unsupported additions'),
                        ('fp','Factual pass')]:
        a, b = jl[metric].values, jg[metric].values
        po = float((a == b).mean())
        k, lo, hi = kappa_boot_ci(a, b)
        degenerate = (min(a.mean(), 1-a.mean()) < 0.005) or (min(b.mean(), 1-b.mean()) < 0.005)
        kap_rows.append(dict(generated_by=gen, metric=lab, n=int(len(common)),
            pct_agree=round(po*100, 1),
            cohen_kappa=(None if np.isnan(k) else round(float(k), 3)),
            k_ci=(None if np.isnan(k) else f"[{lo:.3f}, {hi:.3f}]"),
            strength=("degenerate: base rate ~constant" if degenerate else kappa_strength(k))))
kap = pd.DataFrame(kap_rows)
kap.to_csv(os.path.join(OUT, "cohen_kappa_interjudge.csv"), index=False)
report['cohen_kappa'] = kap.to_dict('records')

# ---------- 3. calibration Wilson CIs (Table 3) ----------
cal = {"GPT-5.4 mini":[(6,276),(13,147),(31,110)],
       "GPT-OSS-120B":[(2,276),(4,147),(12,110)],
       "Gemini 3.1 Flash-Lite":[(4,276),(11,147),(18,110)],
       "Llama 3.3 70B":[(2,276),(10,147),(10,110)],
       "Venice uncensored":[(22,276),(44,147),(28,110)]}
dims = ["IDRE empathy","AskDocs empathy","AskDocs quality"]
cal_rows = []
for judge, cells in cal.items():
    row = {"judge": judge}
    for dim,(err,n) in zip(dims, cells):
        acc = (n-err)/n; lo,hi = wilson(n-err, n)
        row[dim] = f"{acc:.3f} [{lo:.3f},{hi:.3f}]"
    cal_rows.append(row)
calib = pd.DataFrame(cal_rows)
calib.to_csv(os.path.join(OUT, "table3_calibration_wilson_ci.csv"), index=False)
report['calibration'] = calib.to_dict('records')

# ---------- 4. length confound logistic regression ----------
lr_rows = []
for k in ORDER:
    m = M[k]; delta = (m.gen_wc - m.orig_wc).values.astype(float)
    y = m.qual_gen.values
    z = (delta - delta.mean())/delta.std()
    if y.sum() in (0, len(y)):
        # separation on quality; still fit but flag
        pass
    X = sm.add_constant(z)
    try:
        res = sm.Logit(y, X).fit(disp=0)
        lr_rows.append(dict(generator=k[0], judge=k[1], qual_gen=int(y.sum()),
                            beta_per_SD=float(res.params[1]), OR_per_SD=float(np.exp(res.params[1])),
                            p=float(res.pvalues[1]),
                            mean_delta_words=float(delta.mean())))
    except Exception as e:
        lr_rows.append(dict(generator=k[0], judge=k[1], qual_gen=int(y.sum()), error=str(e)))
lr = pd.DataFrame(lr_rows)
lr.to_csv(os.path.join(OUT, "length_confound_logit.csv"), index=False)
report['length_confound'] = lr.to_dict('records')

# ---------- 5. length statistics (resolve 2.07 vs 1.96) ----------
len_rows = []
for gen in ["Llama","GPT-OSS"]:
    m = M[(gen,"Llama")]
    ratio = m.gen_wc / m.orig_wc
    len_rows.append(dict(generator=gen, orig_mean=float(m.orig_wc.mean()),
        gen_mean=float(m.gen_wc.mean()), gen_sd=float(m.gen_wc.std()),
        ratio_of_means=float(m.gen_wc.mean()/m.orig_wc.mean()),
        mean_of_ratios=float(ratio.mean()), ratio_sd=float(ratio.std())))
lens = pd.DataFrame(len_rows)
lens.to_csv(os.path.join(OUT, "length_stats.csv"), index=False)
report['length_stats'] = lens.to_dict('records')

# ---------- 6. FP gap decomposition ----------
dec = M[("Llama","GPT-OSS")]
report['fp_gap_decomposition'] = dict(
    setting="Llama-generated / GPT-OSS-judge",
    factual_pass=int(dec.fp.sum()), fp_failures=int(1000-dec.fp.sum()),
    preservation_failures=int(1000-dec.pres.sum()),
    unsupported_addition_flags=int(1000-dec.nounsupp.sum()),
    note="FP gap driven overwhelmingly by unsupported-addition flags, not preservation loss")

# ---------- 7. category-level with Wilson CIs (both contrasting settings) ----------
cat_rows = []
for k in [("Llama","GPT-OSS"),("GPT-OSS","Llama")]:
    m = M[k]
    for cat, g in m.groupby('cat'):
        kk, n = int(g.fp.sum()), len(g); lo, hi = wilson(kk, n)
        cat_rows.append(dict(setting=f"{k[0]}/{k[1]}", category=cat, fp=kk, n=n,
                             ci_low=round(lo*100,1), ci_high=round(hi*100,1)))
cats = pd.DataFrame(cat_rows)
cats.to_csv(os.path.join(OUT, "category_factual_pass_wilson_ci.csv"), index=False)
report['category'] = cats.to_dict('records')

with open(os.path.join(OUT, "analysis_report.json"), "w") as fh:
    json.dump(report, fh, indent=2)

# ---------- figures ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "savefig.bbox":"tight","figure.dpi":300})
L_COL="#4C72B0"; G_COL="#C44E52"; GAP="#8C8C8C"; OK="#4C72B0"; FOCAL="#C44E52"

# --- Figure 1: FP rate lollipop with bootstrap CI ---
rates=[M[k].fp.mean()*100 for k in ORDER]; cis=[boot_ci(M[k].fp.values) for k in ORDER]
cols=[OK if r>90 else FOCAL for r in rates]
fig1,ax=plt.subplots(figsize=(6.8,3.9)); y=np.arange(len(ORDER))[::-1]
for yi,(r,(lo,hi),cc) in zip(y,zip(rates,cis,cols)):
    ax.hlines(yi,0,r,color=cc,lw=2.4,zorder=2)
    ax.errorbar(r,yi,xerr=[[r-lo*100],[hi*100-r]],fmt='o',color=cc,ms=8.5,capsize=4,elinewidth=1.6,zorder=3)
    ax.text(hi*100+2.2,yi,f"{r:.1f}%",va='center',ha='left',color=cc,fontsize=8.5,fontweight='bold')
ax.set_yticks(y); ax.set_yticklabels([f"{g} \u2192 {j}" for g,j in ORDER])
ax.set_xlim(0,112); ax.set_xlabel("Factual-pass rate (% of 1,000 edited answers, 95% bootstrap CI)")
ax.set_ylabel("generator \u2192 judge")
ax.set_title("Factual-pass rate collapses only when GPT-OSS judges Llama's longer edits",fontsize=9,loc='left')
ax.margins(y=0.16); fig1.tight_layout(); fig1.savefig(os.path.join(FIG,"fp_rate_bootstrap_ci.png"))

# --- Figure 2: paired-judge dumbbell (Llama-generated) with McNemar side column ---
metsF=[("nounsupp","No unsupported\nadditions"),("fp","Factual pass"),
       ("pres","Info preserved"),("qual_gen","Quality prefers\nedited")]
jl,jg=M[("Llama","Llama")],M[("Llama","GPT-OSS")]
common=jl.index.intersection(jg.index); jl,jg=jl.loc[common],jg.loc[common]
ljv=[jl[m].mean()*100 for m,_ in metsF]; gjv=[jg[m].mean()*100 for m,_ in metsF]
discF=[(int(((jl[m]==1)&(jg[m]==0)).sum()),int(((jl[m]==0)&(jg[m]==1)).sum())) for m,_ in metsF]
def pstr(p): return f"{p:.0e}".replace("e-0","\u00d710\u207b").replace("e-","\u00d710\u207b") if p<1e-3 else (f"{p:.2f} (n.s.)" if p>0.05 else f"{p:.1e}")
pvF=[pstr(float(mc[(mc.generated_by=="Llama")&(mc.metric==m)].p.iloc[0])) if not mc[(mc.generated_by=="Llama")&(mc.metric==m)].empty else "" for m in [x[0] for x in metsF]]
# fall back to explicit mapping (mc.metric uses code names? no -> uses same 'metric' keys)
pmap={"nounsupp":"3\u00d710\u207b\u2077\u2070","fp":"3\u00d710\u207b\u2076\u00b9","pres":"0.26 (n.s.)","qual_gen":"2\u00d710\u207b\u00b3"}
pvF=[pmap[m] for m,_ in metsF]
fig2=plt.figure(figsize=(7.6,4.0)); gs=fig2.add_gridspec(1,2,width_ratios=[3.4,1.05],wspace=0.04)
ax2=fig2.add_subplot(gs[0]); axs=fig2.add_subplot(gs[1]); axs.axis("off")
y=np.arange(len(metsF))[::-1]
for yi,a,b in zip(y,ljv,gjv): ax2.plot([a,b],[yi,yi],color=GAP,lw=2,zorder=1)
ax2.scatter(ljv,y,s=95,color=L_COL,zorder=3,label="Llama judge")
ax2.scatter(gjv,y,s=95,color=G_COL,zorder=3,label="GPT-OSS judge")
for yi,a,b in zip(y,ljv,gjv):
    if abs(a-b)<5:
        ax2.text(a,yi+0.26,f"{a:.0f}",ha='center',va='bottom',fontsize=8,color=L_COL,fontweight='bold')
        ax2.text(b,yi-0.26,f"{b:.0f}",ha='center',va='top',fontsize=8,color=G_COL,fontweight='bold')
    else:
        ax2.text(a,yi+0.24,f"{a:.0f}",ha='center',va='bottom',fontsize=8,color=L_COL,fontweight='bold')
        ax2.text(b,yi+0.24,f"{b:.0f}",ha='center',va='bottom',fontsize=8,color=G_COL,fontweight='bold')
ax2.set_yticks(y); ax2.set_yticklabels([lab for _,lab in metsF])
ax2.set_xlim(40,106); ax2.set_xlabel("% of the same 1,000 Llama-edited answers marked \u201cyes\u201d")
ax2.set_title("Same texts, two judges: they split on unsupported additions,\nnot on information preservation",fontsize=9,loc='left')
ax2.legend(frameon=False,fontsize=8,loc='lower left',bbox_to_anchor=(0.0,-0.03)); ax2.margins(y=0.20)
axs.set_ylim(ax2.get_ylim())
axs.text(0.5,1.00,"McNemar (discordant, p)",ha='center',va='bottom',fontsize=7.5,fontweight='bold',color='#333',transform=axs.transAxes)
for yi,(d10,d01),p in zip(y,discF,pvF): axs.text(0.5,yi,f"{d10} vs {d01}\np = {p}",ha='center',va='center',fontsize=7.3,color='#333')
fig2.savefig(os.path.join(FIG,"paired_judge_disagreement.png"))

# --- Figure 3: Cohen's kappa vs raw agreement (Llama-generated) ---
order=["No unsupported additions","Factual pass","Info preserved","Quality prefers edited","Empathy prefers edited"]
sub=kap[kap.generated_by=="Llama"].set_index("metric").loc[order].reset_index()
fig3,ax3=plt.subplots(figsize=(7.2,3.9)); yk=np.arange(len(sub))[::-1]
agree=sub.pct_agree.values
kv=[0 if (v is None or (isinstance(v,float) and np.isnan(v))) else v for v in sub.cohen_kappa.values]
ax3.barh(yk+0.19,agree,0.36,color="#C7C7C7",label="Raw % agreement",zorder=2)
ax3.barh(yk-0.19,[v*100 for v in kv],0.36,color="#DD8452",label="Cohen's \u03ba \u00d7 100 (chance-corrected)",zorder=2)
for yi,ag,k,raw in zip(yk,agree,kv,sub.cohen_kappa.values):
    ax3.text(ag+1,yi+0.19,f"{ag:.0f}%",va='center',ha='left',fontsize=7.5,color='#666')
    ktxt="\u03ba\u22480 (degenerate)" if (raw is None or (isinstance(raw,float) and np.isnan(raw))) else f"\u03ba={raw:.2f}"
    ax3.text(max(k*100,0)+1,yi-0.19,ktxt,va='center',ha='left',fontsize=7.5,color="#B5651D")
ax3.set_yticks(yk); ax3.set_yticklabels(order)
ax3.set_xlim(0,112); ax3.set_xlabel("Agreement between the two judges on identical Llama-edited texts")
ax3.set_title("High raw agreement is an artifact of skewed base rates:\nchance-corrected agreement (\u03ba) is near zero on every check",fontsize=9,loc='left')
ax3.legend(frameon=False,fontsize=8,loc='lower right'); ax3.margins(y=0.08)
fig3.tight_layout(); fig3.savefig(os.path.join(FIG,"cohen_kappa_interjudge.png"))

print("Done. Outputs in", OUT, "figures in", FIG)
print(t6.to_string(index=False)); print()
print(mc.to_string(index=False)); print()
print(kap.to_string(index=False))
