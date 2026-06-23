"""
cns_drugs.py — Neuricular's curated reference database of CNS drugs for Tanimoto similarity analysis
=============================
Curated reference database of CNS drugs for automated Tanimoto similarity analysis.

Organised by therapeutic category. Each entry includes:
  - name, smiles, category, indication, bbb_mechanism
  - moa (mechanism of action) — shown when a match is found

Used by chem_calc.get_cns_tanimoto_panel() to identify which known CNS drugs
a query molecule most structurally resembles, providing pharmacological context
for the similarity results.

SMILES sourced from PubChem canonical SMILES.
"""

CNS_DRUG_DATABASE = [

    # ── Antidepressants ───────────────────────────────────────────────────────
    {
        "name":      "Sertraline",
        "smiles":    "CN[C@H]1CC[C@H](C2=CC=CC=C12)C3=CC(=C(C=C3)Cl)Cl",
        "category":  "Antidepressant",
        "indication":"Major depressive disorder, OCD, PTSD",
        "moa":       "Selective serotonin reuptake inhibitor (SSRI)",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Fluoxetine",
        "smiles":    "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F",
        "category":  "Antidepressant",
        "indication":"Depression, bulimia, OCD",
        "moa":       "SSRI",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Paroxetine",
        "smiles":    "C1CNC[C@H]([C@@H]1C2=CC=C(C=C2)F)COC3=CC4=C(C=C3)OCO4",
        "category":  "Antidepressant",
        "indication":"Depression, panic disorder, social anxiety",
        "moa":       "SSRI",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Escitalopram",
        "smiles":    "CN(C)CCC[C@@]1(C2=C(CO1)C=C(C=C2)C#N)C3=CC=C(C=C3)F",
        "category":  "Antidepressant",
        "indication":"Depression, generalised anxiety disorder",
        "moa":       "SSRI",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Venlafaxine",
        "smiles":    "CN(C)CC(C1=CC=C(C=C1)OC)C2(CCCCC2)O",
        "category":  "Antidepressant",
        "indication":"Depression, anxiety, fibromyalgia",
        "moa":       "Serotonin-norepinephrine reuptake inhibitor (SNRI)",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Duloxetine",
        "smiles":    "CNCC[C@@H](C1=CC=CS1)OC2=CC=CC3=CC=CC=C32",
        "category":  "Antidepressant",
        "indication":"Depression, neuropathic pain, fibromyalgia",
        "moa":       "SNRI",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Amitriptyline",
        "smiles":    "CN(C)CCC=C1C2=CC=CC=C2CCC3=CC=CC=C31",
        "category":  "Antidepressant",
        "indication":"Depression, neuropathic pain, migraine prophylaxis",
        "moa":       "Tricyclic antidepressant; NE/5-HT reuptake inhibitor + anticholinergic",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Imipramine",
        "smiles":    "CN(C)CCCN1C2=CC=CC=C2CCC3=CC=CC=C31",
        "category":  "Antidepressant",
        "indication":"Depression, enuresis, panic disorder",
        "moa":       "Tricyclic antidepressant (TCA)",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Mirtazapine",
        "smiles":    "CN1CCN2C(C1)C3=CC=CC=C3CC4=C2N=CC=C4",
        "category":  "Antidepressant",
        "indication":"Depression, insomnia, appetite stimulation",
        "moa":       "NaSSA; α2-adrenergic antagonist, 5-HT2/3 antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Bupropion",
        "smiles":    "CC(C(=O)C1=CC(=CC=C1)Cl)NC(C)(C)C",
        "category":  "Antidepressant",
        "indication":"Depression, smoking cessation",
        "moa":       "NE/DA reuptake inhibitor (NDRI)",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Antipsychotics ────────────────────────────────────────────────────────
    {
        "name":      "Clozapine",
        "smiles":    "CN1CCN(CC1)C2=NC3=C(C=CC(=C3)Cl)NC4=CC=CC=C42",
        "category":  "Antipsychotic",
        "indication":"Treatment-resistant schizophrenia",
        "moa":       "Atypical antipsychotic; D4/5-HT2A antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Risperidone",
        "smiles":    "CC1=C(C(=O)N2CCCCC2=N1)CCN3CCC(CC3)C4=NOC5=C4C=CC(=C5)F",
        "category":  "Antipsychotic",
        "indication":"Schizophrenia, bipolar disorder",
        "moa":       "Atypical antipsychotic; D2/5-HT2A antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Olanzapine",
        "smiles":    "CC1=CC2=C(S1)NC3=CC=CC=C3N=C2N4CCN(CC4)C",
        "category":  "Antipsychotic",
        "indication":"Schizophrenia, bipolar disorder",
        "moa":       "Atypical antipsychotic; multiple receptor antagonism",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Quetiapine",
        "smiles":    "C1CN(CCN1CCOCCO)C2=NC3=CC=CC=C3SC4=CC=CC=C42",
        "category":  "Antipsychotic",
        "indication":"Schizophrenia, bipolar disorder, depression (adjunct)",
        "moa":       "Atypical antipsychotic; D2/5-HT2 antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Haloperidol",
        "smiles":    "C1CN(CCC1(C2=CC=C(C=C2)Cl)O)CCCC(=O)C3=CC=C(C=C3)F",
        "category":  "Antipsychotic",
        "indication":"Schizophrenia, Tourette syndrome, acute agitation",
        "moa":       "Typical antipsychotic; D2 antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Aripiprazole",
        "smiles":    "C1CC(=O)NC2=C1C=CC(=C2)OCCCCN3CCN(CC3)C4=C(C(=CC=C4)Cl)Cl",
        "category":  "Antipsychotic",
        "indication":"Schizophrenia, bipolar disorder, MDD adjunct",
        "moa":       "Partial D2/D3 agonist, 5-HT2A antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Ziprasidone",
        "smiles":    "C1CN(CCN1CCC2=C(C=C3C(=C2)CC(=O)N3)Cl)C4=NSC5=CC=CC=C54",
        "category":  "Antipsychotic",
        "indication":"Schizophrenia, bipolar mania",
        "moa":       "D2/5-HT2A antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Anxiolytics / Sedatives ───────────────────────────────────────────────
    {
        "name":      "Diazepam",
        "smiles":    "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3",
        "category":  "Anxiolytic",
        "indication":"Anxiety, muscle spasm, alcohol withdrawal, seizures",
        "moa":       "Benzodiazepine; GABAA positive allosteric modulator",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Alprazolam",
        "smiles":    "CC1=NN=C2N1C3=C(C=C(C=C3)Cl)C(=NC2)C4=CC=CC=C4",
        "category":  "Anxiolytic",
        "indication":"Panic disorder, anxiety",
        "moa":       "Benzodiazepine; GABAA PAM",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Lorazepam",
        "smiles":    "C1=CC=C(C(=C1)C2=NC(C(=O)NC3=C2C=C(C=C3)Cl)O)Cl",
        "category":  "Anxiolytic",
        "indication":"Anxiety, status epilepticus, procedural sedation",
        "moa":       "Benzodiazepine; GABAA PAM",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Buspirone",
        "smiles":    "C1CCC2(C1)CC(=O)N(C(=O)C2)CCCCN3CCN(CC3)C4=NC=CC=N4",
        "category":  "Anxiolytic",
        "indication":"Generalised anxiety disorder",
        "moa":       "5-HT1A partial agonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Zolpidem",
        "smiles":    "CC1=CC=C(C=C1)C2=C(N3C=C(C=CC3=N2)C)CC(=O)N(C)C",
        "category":  "Hypnotic",
        "indication":"Insomnia",
        "moa":       "GABAA PAM; BZ1-selective",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Zaleplon",
        "smiles":    "CCN(C1=CC=CC(=C1)C2=CC=NC3=C(C=NN23)C#N)C(=O)C",
        "category":  "Hypnotic",
        "indication":"Insomnia",
        "moa":       "GABAA PAM; BZ1-selective",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Melatonin",
        "smiles":    "CC(=O)NCCC1=CNC2=C1C=C(C=C2)OC",
        "category":  "Hypnotic",
        "indication":"Sleep disorders, jet lag",
        "moa":       "MT1/MT2 melatonin receptor agonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Antiepileptics / Anticonvulsants ──────────────────────────────────────
    {
        "name":      "Valproic acid",
        "smiles":    "CCCC(CCC)C(=O)O",
        "category":  "Antiepileptic",
        "indication":"Epilepsy, bipolar disorder, migraine prophylaxis",
        "moa":       "Na⁺ channel blocker, GABA transaminase inhibitor",
        "bbb_mechanism": "Passive diffusion (MCT contribution reported)",
    },
    {
        "name":      "Carbamazepine",
        "smiles":    "C1=CC=C2C(=C1)C=CC3=CC=CC=C3N2C(=O)N",
        "category":  "Antiepileptic",
        "indication":"Epilepsy, trigeminal neuralgia, bipolar disorder",
        "moa":       "Voltage-gated Na⁺ channel blocker",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Levetiracetam",
        "smiles":    "CC[C@@H](C(=O)N)N1CCCC1=O",
        "category":  "Antiepileptic",
        "indication":"Focal and generalised epilepsy",
        "moa":       "SV2A synaptic vesicle protein modulator",
        "bbb_mechanism": "Active transport (SLC7A5/LAT1-related)",
    },
    {
        "name":      "Gabapentin",
        "smiles":    "C1CCC(CC1)(CC(=O)O)CN",
        "category":  "Antiepileptic",
        "indication":"Epilepsy, neuropathic pain",
        "moa":       "α2δ voltage-gated Ca²⁺ channel subunit modulator",
        "bbb_mechanism": "LAT1 active transporter",
    },
    {
        "name":      "Pregabalin",
        "smiles":    "CC(C)C[C@@H](CC(=O)O)CN",
        "category":  "Antiepileptic",
        "indication":"Neuropathic pain, epilepsy, fibromyalgia, anxiety",
        "moa":       "α2δ voltage-gated Ca²⁺ channel subunit modulator",
        "bbb_mechanism": "LAT1 active transporter",
    },
    {
        "name":      "Phenytoin",
        "smiles":    "C1=CC=C(C=C1)C2(C(=O)NC(=O)N2)C3=CC=CC=C3",
        "category":  "Antiepileptic",
        "indication":"Epilepsy, arrhythmia",
        "moa":       "Voltage-gated Na⁺ channel blocker",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Zonisamide",
        "smiles":    "C1=CC=C2C(=C1)C(=NO2)CS(=O)(=O)N",
        "category":  "Antiepileptic",
        "indication":"Epilepsy, Parkinson's disease",
        "moa":       "Na⁺/T-type Ca²⁺ channel blocker",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Parkinson's / Movement Disorders ─────────────────────────────────────
    {
        "name":      "Levodopa",
        "smiles":    "C1=CC(=C(C=C1C[C@@H](C(=O)O)N)O)O",
        "category":  "Parkinson's",
        "indication":"Parkinson's disease",
        "moa":       "Dopamine precursor",
        "bbb_mechanism": "LAT1 active transporter",
    },
    {
        "name":      "Pramipexole",
        "smiles":    "CCCN[C@H]1CCC2=C(C1)SC(=N2)N",
        "category":  "Parkinson's",
        "indication":"Parkinson's disease, restless legs syndrome",
        "moa":       "D2/D3 dopamine receptor agonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Ropinirole",
        "smiles":    "CCCN(CCC)CCC1=C2CC(=O)NC2=CC=C1",
        "category":  "Parkinson's",
        "indication":"Parkinson's disease, restless legs syndrome",
        "moa":       "D2/D3 dopamine receptor agonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Selegiline",
        "smiles":    "C[C@H](CC1=CC=CC=C1)N(C)CC#C",
        "category":  "Parkinson's",
        "indication":"Parkinson's disease",
        "moa":       "Irreversible MAO-B inhibitor",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Entacapone",
        "smiles":    "CCN(CC)C(=O)/C(=C/C1=CC(=C(C(=C1)O)O)[N+](=O)[O-])/C#N",
        "category":  "Parkinson's",
        "indication":"Parkinson's disease (adjunct to levodopa)",
        "moa":       "Peripheral COMT inhibitor",
        "bbb_mechanism": "Does not cross BBB (peripheral action)",
    },

    # ── Alzheimer's / Dementia ────────────────────────────────────────────────
    {
        "name":      "Donepezil",
        "smiles":    "COC1=C(C=C2C(=C1)CC(C2=O)CC3CCN(CC3)CC4=CC=CC=C4)OC",
        "category":  "Alzheimer's",
        "indication":"Alzheimer's disease",
        "moa":       "Acetylcholinesterase (AChE) inhibitor",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Memantine",
        "smiles":    "CC12CC3CC(C1)(CC(C3)(C2)N)C",
        "category":  "Alzheimer's",
        "indication":"Moderate–severe Alzheimer's disease",
        "moa":       "NMDA receptor antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Rivastigmine",
        "smiles":    "CCN(C)C(=O)OC1=CC=CC(=C1)[C@H](C)N(C)C",
        "category":  "Alzheimer's",
        "indication":"Alzheimer's disease, Parkinson's dementia",
        "moa":       "Pseudo-irreversible AChE/BuChE inhibitor",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Galantamine",
        "smiles":    "CN1CC[C@@]23C=C[C@@H](C[C@@H]2OC4=C(C=CC(=C34)C1)OC)O",
        "category":  "Alzheimer's",
        "indication":"Mild–moderate Alzheimer's disease",
        "moa":       "AChE inhibitor + nicotinic receptor PAM",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Opioid Analgesics ─────────────────────────────────────────────────────
    {
        "name":      "Morphine",
        "smiles":    "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O",
        "category":  "Opioid analgesic",
        "indication":"Severe pain",
        "moa":       "μ-opioid receptor agonist",
        "bbb_mechanism": "Partial passive diffusion; P-gp efflux limits CNS exposure",
    },
    {
        "name":      "Oxycodone",
        "smiles":    "CN1CC[C@]23[C@@H]4C(=O)CC[C@]2([C@H]1CC5=C3C(=C(C=C5)OC)O4)O",
        "category":  "Opioid analgesic",
        "indication":"Moderate–severe pain",
        "moa":       "μ-opioid receptor agonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Tramadol",
        "smiles":    "CN(C)C[C@H]1CCCC[C@@]1(C2=CC(=CC=C2)OC)O",
        "category":  "Opioid analgesic",
        "indication":"Moderate–severe pain",
        "moa":       "Weak μ-opioid agonist + SNRI",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Fentanyl",
        "smiles":    "CCC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3",
        "category":  "Opioid analgesic",
        "indication":"Severe pain, anaesthesia",
        "moa":       "μ-opioid receptor agonist",
        "bbb_mechanism": "Rapid passive transcellular diffusion (high lipophilicity)",
    },
    {
        "name":      "Buprenorphine",
        "smiles":    "C[C@]([C@H]1C[C@@]23CC[C@@]1([C@H]4[C@@]25CCN([C@@H]3CC6=C5C(=C(C=C6)O)O4)CC7CC7)OC)(C(C)(C)C)O",
        "category":  "Opioid analgesic",
        "indication":"Pain, opioid use disorder",
        "moa":       "Partial μ-opioid agonist / κ-antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Naloxone",
        "smiles":    "C=CCN1CC[C@]23[C@@H]4C(=O)CC[C@]2([C@H]1CC5=C3C(=C(C=C5)O)O4)O",
        "category":  "Opioid antagonist",
        "indication":"Opioid overdose reversal",
        "moa":       "μ-opioid receptor antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Non-opioid Analgesics / Migraine ──────────────────────────────────────
    {
        "name":      "Sumatriptan",
        "smiles":    "CNS(=O)(=O)CC1=CC2=C(C=C1)NC=C2CCN(C)C",
        "category":  "Migraine",
        "indication":"Acute migraine",
        "moa":       "5-HT1B/1D agonist; triptan",
        "bbb_mechanism": "Low CNS penetration (peripheral cranial vasculature action)",
    },
    {
        "name":      "Rizatriptan",
        "smiles":    "CN(C)CCC1=CNC2=C1C=C(C=C2)CN3C=NC=N3",
        "category":  "Migraine",
        "indication":"Acute migraine",
        "moa":       "5-HT1B/1D agonist",
        "bbb_mechanism": "Partial passive diffusion",
    },
    {
        "name":      "Topiramate",
        "smiles":    "CC1(O[C@@H]2CO[C@@]3([C@H]([C@@H]2O1)OC(O3)(C)C)COS(=O)(=O)N)C",
        "category":  "Migraine, Epilepsy",
        "indication":"Migraine prophylaxis, Epilepsy",
        "moa":       "Na⁺ channel blocker, AMPA antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── ADHD ──────────────────────────────────────────────────────────────────
    {
        "name":      "Methylphenidate",
        "smiles":    "COC(=O)C(C1CCCCN1)C2=CC=CC=C2",
        "category":  "ADHD",
        "indication":"ADHD, narcolepsy",
        "moa":       "DAT/NET reuptake inhibitor",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Atomoxetine",
        "smiles":    "CC1=CC=CC=C1O[C@H](CCNC)C2=CC=CC=C2",
        "category":  "ADHD",
        "indication":"ADHD",
        "moa":       "Selective NET reuptake inhibitor",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Lisdexamfetamine",
        "smiles":    "C[C@@H](CC1=CC=CC=C1)NC(=O)[C@H](CCCCN)N",
        "category":  "ADHD",
        "indication":"ADHD, binge eating disorder",
        "moa":       "Prodrug of d-amphetamine; DAT/NET releaser",
        "bbb_mechanism": "Hydrolysed to amphetamine; amphetamine crosses by passive diffusion",
    },

    # ── Multiple Sclerosis ────────────────────────────────────────────────────
    {
        "name":      "Baclofen",
        "smiles":    "C1=CC(=CC=C1C(CC(=O)O)CN)Cl",
        "category":  "Multiple sclerosis / spasticity",
        "indication":"Spasticity (MS, spinal injury)",
        "moa":       "GABAB receptor agonist",
        "bbb_mechanism": "Passive transcellular diffusion (limited; mostly spinal)",
    },
    {
        "name":      "Tizanidine",
        "smiles":    "C1CN=C(N1)NC2=C(C=CC3=NSN=C32)Cl",
        "category":  "Multiple sclerosis / spasticity",
        "indication":"Spasticity",
        "moa":       "α2-adrenergic agonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    # ── General Anaesthetics / Sedatives ──────────────────────────────────────
    {
        "name":      "Propofol",
        "smiles":    "CC(C)C1=C(C(=CC=C1)C(C)C)O",
        "category":  "Anaesthetic",
        "indication":"General anaesthesia, procedural sedation",
        "moa":       "GABAA PAM (potentiates Cl⁻ current)",
        "bbb_mechanism": "Rapid passive diffusion (highly lipophilic)",
    },
    {
        "name":      "Ketamine",
        "smiles":    "CNC1(CCCCC1=O)C2=CC=CC=C2Cl",
        "category":  "Anaesthetic",
        "indication":"Anaesthesia, treatment-resistant depression",
        "moa":       "NMDA receptor antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Midazolam",
        "smiles":    "CC1=NC=C2N1C3=C(C=C(C=C3)Cl)C(=NC2)C4=CC=CC=C4F",
        "category":  "Anaesthetic",
        "indication":"Procedural sedation, status epilepticus",
        "moa":       "Benzodiazepine; GABAA PAM",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Dexmedetomidine",
        "smiles":    "CC1=C(C(=CC=C1)[C@H](C)C2=CN=CN2)C",
        "category":  "Anaesthetic",
        "indication":"ICU sedation",
        "moa":       "α2-adrenergic agonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Stimulants  ───────────────────────────────────────────────────────
    {
        "name":      "Caffeine",
        "smiles":    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "category":  "Stimulant",
        "indication":"CNS stimulant, apnoea of prematurity",
        "moa":       "Adenosine A1/A2A receptor antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Modafinil",
        "smiles":    "C1=CC=C(C=C1)C(C2=CC=CC=C2)S(=O)CC(=O)N",
        "category":  "Stimulant",
        "indication":"Narcolepsy, shift work disorder",
        "moa":       "DAT inhibitor / wakefulness-promoting (exact MOA unclear)",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Mood Stabilisers ──────────────────────────────────────────────────────
    {
        "name":      "Lithium carbonate",
        "smiles":    "[Li+].[Li+].C(=O)([O-])[O-]",
        "category":  "Mood stabiliser",
        "indication":"Bipolar disorder, cluster headache",
        "moa":       "GSK-3β inhibitor, inositol depletion",
        "bbb_mechanism": "Ion channel–mediated (Li⁺ ion transport)",
    },
    {
        "name":      "Lamotrigine",
        "smiles":    "C1=CC(=C(C(=C1)Cl)Cl)C2=C(N=C(N=N2)N)N",
        "category":  "Mood stabiliser",
        "indication":"Bipolar disorder, epilepsy",
        "moa":       "Na⁺ channel blocker",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Addiction / Substance Use ─────────────────────────────────────────────
    {
        "name":      "Naltrexone",
        "smiles":    "C1CC1CN2CC[C@]34[C@@H]5C(=O)CC[C@]3([C@H]2CC6=C4C(=C(C=C6)O)O5)O",
        "category":  "Addiction",
        "indication":"Opioid/alcohol use disorder",
        "moa":       "μ-opioid receptor antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Varenicline",
        "smiles":    "C1[C@@H]2CNC[C@H]1C3=CC4=NC=CN=C4C=C23",
        "category":  "Addiction",
        "indication":"Smoking cessation",
        "moa":       "Partial α4β2 nicotinic ACh receptor agonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Acamprosate",
        "smiles":    "CC(=O)NCCCS(=O)(=O)O",
        "category":  "Addiction",
        "indication":"Alcohol use disorder",
        "moa":       "NMDA receptor modulator / GABAA agonist",
        "bbb_mechanism": "Passive transcellular diffusion (limited)",
    },

    # ── Headache / CNS Vascular ───────────────────────────────────────────────
    {
        "name":      "Ergotamine",
        "smiles":    "C[C@@]1(C(=O)N2[C@H](C(=O)N3CCC[C@H]3[C@@]2(O1)O)CC4=CC=CC=C4)NC(=O)[C@H]5CN([C@@H]6CC7=CNC8=CC=CC(=C78)C6=C5)C",
        "category":  "Migraine",
        "indication":"Acute migraine, cluster headache",
        "moa":       "5-HT1B/1D agonist + partial agonist at α-adrenergic receptors",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Spinal cord / muscle relaxants ───────────────────────────────────────
    {
        "name":      "Cyclobenzaprine",
        "smiles":    "CN(C)CCC=C1C2=CC=CC=C2C=CC3=CC=CC=C31",
        "category":  "Muscle relaxant",
        "indication":"Muscle spasm",
        "moa":       "Central α1/5-HT2 antagonist (structurally TCA-like)",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Carisoprodol",
        "smiles":    "CCCC(C)(COC(=O)N)COC(=O)NC(C)C",
        "category":  "Muscle relaxant",
        "indication":"Acute musculoskeletal pain",
        "moa":       "GABAA PAM (metabolised to meprobamate)",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Cognitive enhancers / Nootropics ─────────────────────────────────────
    {
        "name":      "Piracetam",
        "smiles":    "C1CC(=O)N(C1)CC(=O)N",
        "category":  "Nootropic",
        "indication":"Cognitive impairment, myoclonus",
        "moa":       "Racetam; AMPA receptor modulator (exact MOA unclear)",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── ALS (Amyotrophic Lateral Sclerosis) ──────────────────────────────────
    {
        "name":      "Riluzole",
        "smiles":    "C1=CC2=C(C=C1OC(F)(F)F)SC(=N2)N",
        "category":  "ALS",
        "indication":"Amyotrophic lateral sclerosis; extends survival",
        "moa":       "Glutamate release inhibitor; Na⁺ channel blocker; NMDA antagonist",
        "bbb_mechanism": "Passive transcellular diffusion (high lipophilicity)",
    },
    {
        "name":      "Edaravone",
        "smiles":    "[2H]C1=C(C(=C(C(=C1[2H])[2H])N2C(=O)CC(=N2)C)[2H])[2H]",
        "category":  "ALS",
        "indication":"ALS; free radical scavenger (approved in Japan/US)",
        "moa":       "Reactive oxygen species scavenger",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── SMA (Spinal Muscular Atrophy) ─────────────────────────────────────────
    {
        "name":      "Risdiplam",
        "smiles":    "CC1=CC(=NN2C1=NC(=C2)C)C3=CC(=O)N4C=C(C=CC4=N3)N5CCNC6(C5)CC6",
        "category":  "SMA",
        "indication":"Spinal muscular atrophy; oral SMN2 splicing modifier",
        "moa":       "SMN2 pre-mRNA splicing modifier; increases full-length SMN protein",
        "bbb_mechanism": "Passive transcellular diffusion; designed for CNS penetration",
    },

    # ── Huntington's Disease ──────────────────────────────────────────────────
    {
        "name":      "Tetrabenazine",
        "smiles":    "CC(C)CC1CN2CCC3=CC(=C(C=C3C2CC1=O)OC)OC",
        "category":  "Huntington's",
        "indication":"Huntington's disease chorea, tardive dyskinesia",
        "moa":       "VMAT2 inhibitor; depletes presynaptic monoamine stores",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Deutetrabenazine",
        "smiles":    "[2H]C([2H])([2H])OC1=C(C=C2[C@@H]3CC(=O)[C@H](CN3CCC2=C1)CC(C)C)OC([2H])([2H])[2H]",
        "category":  "Huntington's",
        "indication":"Huntington's disease chorea, tardive dyskinesia",
        "moa":       "VMAT2 inhibitor; deuterated tetrabenazine analogue (improved PK)",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Multiple Sclerosis (additional) ──────────────────────────────────────
    {
        "name":      "Dimethyl fumarate",
        "smiles":    "COC(=O)/C=C/C(=O)OC",
        "category":  "Multiple sclerosis",
        "indication":"Relapsing-remitting MS",
        "moa":       "Nrf2 pathway activator; anti-inflammatory, neuroprotective",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Siponimod",
        "smiles":    "CCC1=C(C=CC(=C1)/C(=N/OCC2=CC(=C(C=C2)C3CCCCC3)C(F)(F)F)/C)CN4CC(C4)C(=O)O",
        "category":  "Multiple sclerosis",
        "indication":"Secondary progressive MS",
        "moa":       "S1P1/S1P5 receptor modulator; sequesters lymphocytes in lymph nodes",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Narcolepsy / Wakefulness disorders ───────────────────────────────────
    {
        "name":      "Sodium oxybate",
        "smiles":    "C(CC(=O)[O-])CO.[Na+]",
        "category":  "Narcolepsy",
        "indication":"Narcolepsy with cataplexy",
        "moa":       "GABAB agonist / GHB receptor agonist",
        "bbb_mechanism": "MCT1-mediated active transport",
    },
    {
        "name":      "Pitolisant",
        "smiles":    "C1CCN(CC1)CCCOCCCC2=CC=C(C=C2)Cl",
        "category":  "Narcolepsy",
        "indication":"Narcolepsy",
        "moa":       "Histamine H3 receptor inverse agonist/antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Vertigo / Vestibular ──────────────────────────────────────────────────
    {
        "name":      "Betahistine",
        "smiles":    "CNCCC1=CC=CC=N1",
        "category":  "Vestibular",
        "indication":"Ménière's disease, vertigo",
        "moa":       "H1 agonist / H3 antagonist; improves labyrinthine blood flow",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Neuroprotection / Cerebrovascular ────────────────────────────────────
    {
        "name":      "Nimodipine",
        "smiles":    "CC1=C(C(C(=C(N1)C)C(=O)OC(C)C)C2=CC(=CC=C2)[N+](=O)[O-])C(=O)OCCOC",
        "category":  "Cerebrovascular",
        "indication":"Subarachnoid haemorrhage; prevents vasospasm",
        "moa":       "L-type Ca²⁺ channel blocker (CNS-selective)",
        "bbb_mechanism": "Passive transcellular diffusion (high lipophilicity)",
    },
    {
        "name":      "Vinpocetine",
        "smiles":    "CC[C@@]12CCCN3[C@@H]1C4=C(CC3)C5=CC=CC=C5N4C(=C2)C(=O)OCC",
        "category":  "Cerebrovascular",
        "indication":"Cognitive impairment, cerebrovascular disorders",
        "moa":       "PDE1 inhibitor; Na⁺ channel blocker; cerebral vasodilator",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Nausea / CNS antiemetics ──────────────────────────────────────────────
    {
        "name":      "Ondansetron",
        "smiles":    "CC1=NC=CN1CC2CCC3=C(C2=O)C4=CC=CC=C4N3C",
        "category":  "Antiemetic",
        "indication":"Chemotherapy-induced nausea, post-operative nausea",
        "moa":       "5-HT3 receptor antagonist",
        "bbb_mechanism": "Passive transcellular diffusion (limited CNS penetration)",
    },
    {
        "name":      "Metoclopramide",
        "smiles":    "CCN(CC)CCNC(=O)C1=CC(=C(C=C1OC)N)Cl",
        "category":  "Antiemetic",
        "indication":"Nausea, gastroparesis",
        "moa":       "D2 antagonist / 5-HT4 agonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Peripherally-acting controls (intentionally poor CNS) ────────────────
    {
        "name":      "Atenolol",
        "smiles":    "CC(C)NCC(COC1=CC=C(C=C1)CC(=O)N)O",
        "category":  "Peripheral control",
        "indication":"Hypertension (peripheral beta-blocker; low CNS penetration)",
        "moa":       "β1-adrenergic antagonist",
        "bbb_mechanism": "Excluded by high polarity and P-gp efflux",
    },
    {
        "name":      "Neostigmine",
        "smiles":    "CN(C)C(=O)OC1=CC=CC(=C1)[N+](C)(C)C",
        "category":  "Peripheral control",
        "indication":"Myasthenia gravis, reversal of NMB (peripheral)",
        "moa":       "Quaternary AChE inhibitor; does not cross BBB",
        "bbb_mechanism": "Quaternary ammonium; cannot cross BBB",
    },
]
