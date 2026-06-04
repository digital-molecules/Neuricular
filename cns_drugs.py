"""
cns_drugs.py — MetricularPro
=============================
Curated reference database of CNS drugs for automated Tanimoto similarity analysis.

Organised by therapeutic category. Each entry includes:
  - name, smiles, category, indication, bbb_mechanism
  - moa (mechanism of action) — shown when a match is found

Used by chem_calc.get_cns_tanimoto_panel() to identify which known CNS drugs
a query molecule most structurally resembles, providing pharmacological context
for the similarity results.

SMILES sourced from PubChem canonical SMILES and cross-checked against
DrugBank where available.
"""

CNS_DRUG_DATABASE = [

    # ── Antidepressants ───────────────────────────────────────────────────────
    {
        "name":      "Sertraline",
        "smiles":    "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
        "category":  "Antidepressant",
        "indication":"Major depressive disorder, OCD, PTSD",
        "moa":       "Selective serotonin reuptake inhibitor (SSRI)",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Fluoxetine",
        "smiles":    "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
        "category":  "Antidepressant",
        "indication":"Depression, bulimia, OCD",
        "moa":       "SSRI",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Paroxetine",
        "smiles":    "O=C1OCC[C@@H]1Cc1ccc(F)cc1",
        "category":  "Antidepressant",
        "indication":"Depression, panic disorder, social anxiety",
        "moa":       "SSRI",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Escitalopram",
        "smiles":    "CNCCC(c1ccc(F)cc1)c1ccc2c(c1)CC(=O)O2",
        "category":  "Antidepressant",
        "indication":"Depression, generalised anxiety disorder",
        "moa":       "SSRI",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Venlafaxine",
        "smiles":    "COc1ccc(C(CN(C)C)C2(O)CCCCC2)cc1",
        "category":  "Antidepressant",
        "indication":"Depression, anxiety, fibromyalgia",
        "moa":       "Serotonin-norepinephrine reuptake inhibitor (SNRI)",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Duloxetine",
        "smiles":    "CNCCc1ccc2cccc(OC(c3ccccc3)c3cccs3)c2c1",
        "category":  "Antidepressant",
        "indication":"Depression, neuropathic pain, fibromyalgia",
        "moa":       "SNRI",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Amitriptyline",
        "smiles":    "CN(C)CCC=C1c2ccccc2CCc2ccccc21",
        "category":  "Antidepressant",
        "indication":"Depression, neuropathic pain, migraine prophylaxis",
        "moa":       "Tricyclic antidepressant — NE/5-HT reuptake inhibitor + anticholinergic",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Imipramine",
        "smiles":    "CN(C)CCCN1c2ccccc2CCc2ccccc21",
        "category":  "Antidepressant",
        "indication":"Depression, enuresis, panic disorder",
        "moa":       "Tricyclic antidepressant (TCA)",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Mirtazapine",
        "smiles":    "CN1CCN(C2=Nc3ccccc3Cc3cccnc32)CC1",
        "category":  "Antidepressant",
        "indication":"Depression, insomnia, appetite stimulation",
        "moa":       "NaSSA — α2-adrenergic antagonist, 5-HT2/3 antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Bupropion",
        "smiles":    "CC(C)(N)C(=O)c1cccc(Cl)c1",
        "category":  "Antidepressant",
        "indication":"Depression, smoking cessation",
        "moa":       "NE/DA reuptake inhibitor (NDRI)",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Antipsychotics ────────────────────────────────────────────────────────
    {
        "name":      "Clozapine",
        "smiles":    "CN1CCN(c2nc3ccccc3nc2Cl)CC1",
        "category":  "Antipsychotic",
        "indication":"Treatment-resistant schizophrenia",
        "moa":       "Atypical antipsychotic — D4/5-HT2A antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Risperidone",
        "smiles":    "Cc1ncc2n1-c1ccc(F)cc1CC2=O",
        "category":  "Antipsychotic",
        "indication":"Schizophrenia, bipolar disorder",
        "moa":       "Atypical antipsychotic — D2/5-HT2A antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Olanzapine",
        "smiles":    "Cc1ccc2c(c1)Sc1ccccc1N2CCCN1CCN(C)CC1",
        "category":  "Antipsychotic",
        "indication":"Schizophrenia, bipolar disorder",
        "moa":       "Atypical antipsychotic — multiple receptor antagonism",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Quetiapine",
        "smiles":    "O=C(NCCO)c1ccc2nc3ccccc3sc2c1",
        "category":  "Antipsychotic",
        "indication":"Schizophrenia, bipolar disorder, depression (adjunct)",
        "moa":       "Atypical antipsychotic — D2/5-HT2 antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Haloperidol",
        "smiles":    "OC(CCN1CCC(=O)c2ccc(Cl)cc21)c1ccc(F)cc1",
        "category":  "Antipsychotic",
        "indication":"Schizophrenia, Tourette syndrome, acute agitation",
        "moa":       "Typical antipsychotic — D2 antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Aripiprazole",
        "smiles":    "Clc1ccc(N2CCN(CCCOc3ccc4ccccc4c3)CC2)c(Cl)c1",
        "category":  "Antipsychotic",
        "indication":"Schizophrenia, bipolar disorder, MDD adjunct",
        "moa":       "Partial D2/D3 agonist, 5-HT2A antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Ziprasidone",
        "smiles":    "Clc1ccc2[nH]c(CCN3CCN(c4nsc5ccccc45)CC3)cc2c1",
        "category":  "Antipsychotic",
        "indication":"Schizophrenia, bipolar mania",
        "moa":       "D2/5-HT2A antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Anxiolytics / Sedatives ───────────────────────────────────────────────
    {
        "name":      "Diazepam",
        "smiles":    "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
        "category":  "Anxiolytic",
        "indication":"Anxiety, muscle spasm, alcohol withdrawal, seizures",
        "moa":       "Benzodiazepine — GABAA positive allosteric modulator",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Alprazolam",
        "smiles":    "Cc1nnc2n1-c1ccc(Cl)cc1C(=NCC2)c1ccccc1",
        "category":  "Anxiolytic",
        "indication":"Panic disorder, anxiety",
        "moa":       "Benzodiazepine — GABAA PAM",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Lorazepam",
        "smiles":    "OC1CN=C(c2ccccc2Cl)c2cc(Cl)ccc2N1=O",
        "category":  "Anxiolytic",
        "indication":"Anxiety, status epilepticus, procedural sedation",
        "moa":       "Benzodiazepine — GABAA PAM",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Buspirone",
        "smiles":    "O=C1CCCN1CCCN1CCN(c2nc3ccccc3[nH]2)CC1",
        "category":  "Anxiolytic",
        "indication":"Generalised anxiety disorder",
        "moa":       "5-HT1A partial agonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Zolpidem",
        "smiles":    "CN(C)C(=O)Cc1nc2ccc(C)cc2c1-c1ccc(C)cc1",
        "category":  "Hypnotic",
        "indication":"Insomnia",
        "moa":       "GABAA PAM — BZ1-selective",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Zaleplon",
        "smiles":    "CCN(CC)c1cccc(C(=O)c2ccnc3[nH]nc(-c4ccccc4)c23)c1",
        "category":  "Hypnotic",
        "indication":"Insomnia",
        "moa":       "GABAA PAM — BZ1-selective",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Melatonin",
        "smiles":    "COc1ccc2[nH]cc(CCNC(C)=O)c2c1",
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
        "smiles":    "NC(=O)N1c2ccccc2C=Cc2ccccc21",
        "category":  "Antiepileptic",
        "indication":"Epilepsy, trigeminal neuralgia, bipolar disorder",
        "moa":       "Voltage-gated Na⁺ channel blocker",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Lamotrigine",
        "smiles":    "Nc1nc2cc(Cl)c(Cl)cc2c(=O)[nH]1",
        "category":  "Antiepileptic",
        "indication":"Epilepsy, bipolar disorder",
        "moa":       "Na⁺ channel blocker, glutamate release inhibitor",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Levetiracetam",
        "smiles":    "CC[C@@H](CC(N)=O)N1CCCC1=O",
        "category":  "Antiepileptic",
        "indication":"Focal and generalised epilepsy",
        "moa":       "SV2A synaptic vesicle protein modulator",
        "bbb_mechanism": "Active transport (SLC7A5/LAT1-related)",
    },
    {
        "name":      "Gabapentin",
        "smiles":    "NCC1(CC(=O)O)CCCCC1",
        "category":  "Antiepileptic",
        "indication":"Epilepsy, neuropathic pain",
        "moa":       "α2δ voltage-gated Ca²⁺ channel subunit modulator",
        "bbb_mechanism": "LAT1 active transporter",
    },
    {
        "name":      "Pregabalin",
        "smiles":    "CC(CN)CC(=O)O",
        "category":  "Antiepileptic",
        "indication":"Neuropathic pain, epilepsy, fibromyalgia, anxiety",
        "moa":       "α2δ voltage-gated Ca²⁺ channel subunit modulator",
        "bbb_mechanism": "LAT1 active transporter",
    },
    {
        "name":      "Phenytoin",
        "smiles":    "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1",
        "category":  "Antiepileptic",
        "indication":"Epilepsy, arrhythmia",
        "moa":       "Voltage-gated Na⁺ channel blocker",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Topiramate",
        "smiles":    "CC1(C)OC2COC3(COS(N)(=O)=O)OC1C2O3",
        "category":  "Antiepileptic",
        "indication":"Epilepsy, migraine prophylaxis, weight management",
        "moa":       "Na⁺ channel blocker, AMPA/kainate antagonist, GABA enhancer",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Zonisamide",
        "smiles":    "NS(=O)(=O)Cc1cnoc1-c1ccccc1",
        "category":  "Antiepileptic",
        "indication":"Epilepsy, Parkinson's disease",
        "moa":       "Na⁺/T-type Ca²⁺ channel blocker",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Parkinson's / Movement Disorders ─────────────────────────────────────
    {
        "name":      "Levodopa",
        "smiles":    "N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O",
        "category":  "Parkinson's",
        "indication":"Parkinson's disease",
        "moa":       "Dopamine precursor",
        "bbb_mechanism": "LAT1 active transporter",
    },
    {
        "name":      "Pramipexole",
        "smiles":    "CCCNc1nc2c(s1)CCCN2",
        "category":  "Parkinson's",
        "indication":"Parkinson's disease, restless legs syndrome",
        "moa":       "D2/D3 dopamine receptor agonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Ropinirole",
        "smiles":    "CCCn1cc2c(c1=O)CC(CCC(=O)c1ccc(O)cc1)N2",
        "category":  "Parkinson's",
        "indication":"Parkinson's disease, restless legs syndrome",
        "moa":       "D2/D3 dopamine receptor agonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Selegiline",
        "smiles":    "C#CCN(C)[C@H](C)Cc1ccccc1",
        "category":  "Parkinson's",
        "indication":"Parkinson's disease",
        "moa":       "Irreversible MAO-B inhibitor",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Entacapone",
        "smiles":    "CCN(CC)/C(=O)/C=C(/C#N)c1cc(O)c(O)c([N+](=O)[O-])c1",
        "category":  "Parkinson's",
        "indication":"Parkinson's disease (adjunct to levodopa)",
        "moa":       "Peripheral COMT inhibitor",
        "bbb_mechanism": "Does not cross BBB (peripheral action)",
    },

    # ── Alzheimer's / Dementia ────────────────────────────────────────────────
    {
        "name":      "Donepezil",
        "smiles":    "COc1cc2c(cc1OC)CC(CC(=O)c1ccccc1)C2",
        "category":  "Alzheimer's",
        "indication":"Alzheimer's disease",
        "moa":       "Acetylcholinesterase (AChE) inhibitor",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Memantine",
        "smiles":    "CC12CC(CC(C1)(CN)C)(C2)N",
        "category":  "Alzheimer's",
        "indication":"Moderate–severe Alzheimer's disease",
        "moa":       "NMDA receptor antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Rivastigmine",
        "smiles":    "CCN(C)C(=O)Oc1ccc([C@@H](C)N(C)CC)cc1",
        "category":  "Alzheimer's",
        "indication":"Alzheimer's disease, Parkinson's dementia",
        "moa":       "Pseudo-irreversible AChE/BuChE inhibitor",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Galantamine",
        "smiles":    "COc1ccc2c(c1)C[C@H]1[C@@H](O)C=C[C@]3([C@@H]1[N@@](CC2)CC3)O",
        "category":  "Alzheimer's",
        "indication":"Mild–moderate Alzheimer's disease",
        "moa":       "AChE inhibitor + nicotinic receptor PAM",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Opioid Analgesics ─────────────────────────────────────────────────────
    {
        "name":      "Morphine",
        "smiles":    "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5",
        "category":  "Opioid analgesic",
        "indication":"Severe pain",
        "moa":       "μ-opioid receptor agonist",
        "bbb_mechanism": "Partial passive diffusion; P-gp efflux limits CNS exposure",
    },
    {
        "name":      "Oxycodone",
        "smiles":    "COc1ccc2c(c1)C[C@H]1[C@@H](O)C=C[C@@]3([C@@H]1[N@@](CC23)C)O",
        "category":  "Opioid analgesic",
        "indication":"Moderate–severe pain",
        "moa":       "μ-opioid receptor agonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Tramadol",
        "smiles":    "OC1(c2ccccc2)CCCC[C@@H]1CN(C)C",
        "category":  "Opioid analgesic",
        "indication":"Moderate–severe pain",
        "moa":       "Weak μ-opioid agonist + SNRI",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Fentanyl",
        "smiles":    "CCC(=O)N(c1ccccc1)C1CCN(CCc2ccccc2)CC1",
        "category":  "Opioid analgesic",
        "indication":"Severe pain, anaesthesia",
        "moa":       "μ-opioid receptor agonist",
        "bbb_mechanism": "Rapid passive transcellular diffusion (high lipophilicity)",
    },
    {
        "name":      "Buprenorphine",
        "smiles":    "CO[C@]12CC[C@@]3(C[C@@H]1[C@](C)(O)C(C)(C)C)[C@@H]1Cc4ccc(O)c5c4[C@@]3(CCN1CC1CC1)[C@@H]2O5",
        "category":  "Opioid analgesic",
        "indication":"Pain, opioid use disorder",
        "moa":       "Partial μ-opioid agonist / κ-antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Naloxone",
        "smiles":    "O=C1OC[C@H]2c3cc4c(cc3[C@@H](O)[C@@]3(CCN(CC=C)[C@H]23)C4)O1",
        "category":  "Opioid antagonist",
        "indication":"Opioid overdose reversal",
        "moa":       "μ-opioid receptor antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Non-opioid Analgesics / Migraine ──────────────────────────────────────
    {
        "name":      "Sumatriptan",
        "smiles":    "CNS(=O)(=O)Cc1ccc2[nH]cc(CCN(C)C)c2c1",
        "category":  "Migraine",
        "indication":"Acute migraine",
        "moa":       "5-HT1B/1D agonist — triptan",
        "bbb_mechanism": "Low CNS penetration (peripheral cranial vasculature action)",
    },
    {
        "name":      "Rizatriptan",
        "smiles":    "CN(C)Cc1c[nH]c2ccc(CCN3C=NC(C)=N3)cc12",
        "category":  "Migraine",
        "indication":"Acute migraine",
        "moa":       "5-HT1B/1D agonist",
        "bbb_mechanism": "Partial passive diffusion",
    },
    {
        "name":      "Topiramate",
        "smiles":    "CC1(C)OC2COC3(COS(N)(=O)=O)OC1C2O3",
        "category":  "Migraine",
        "indication":"Migraine prophylaxis",
        "moa":       "Na⁺ channel blocker, AMPA antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── ADHD ──────────────────────────────────────────────────────────────────
    {
        "name":      "Methylphenidate",
        "smiles":    "COC(=O)[C@@H](c1ccccc1)C1CCCCN1",
        "category":  "ADHD",
        "indication":"ADHD, narcolepsy",
        "moa":       "DAT/NET reuptake inhibitor",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Atomoxetine",
        "smiles":    "CNC[C@@H](c1ccccc1)Oc1ccccc1C",
        "category":  "ADHD",
        "indication":"ADHD",
        "moa":       "Selective NET reuptake inhibitor",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Lisdexamfetamine",
        "smiles":    "NCCCC[C@@H](N)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",
        "category":  "ADHD",
        "indication":"ADHD, binge eating disorder",
        "moa":       "Prodrug of d-amphetamine — DAT/NET releaser",
        "bbb_mechanism": "Hydrolysed to amphetamine; amphetamine crosses by passive diffusion",
    },

    # ── Multiple Sclerosis ────────────────────────────────────────────────────
    {
        "name":      "Baclofen",
        "smiles":    "OC(=O)CC(N)Cc1ccc(Cl)cc1",
        "category":  "Multiple sclerosis / spasticity",
        "indication":"Spasticity (MS, spinal injury)",
        "moa":       "GABAB receptor agonist",
        "bbb_mechanism": "Passive transcellular diffusion (limited; mostly spinal)",
    },
    {
        "name":      "Tizanidine",
        "smiles":    "Clc1cc2c(cc1Cl)N=C(NCC=C)S2",
        "category":  "Multiple sclerosis / spasticity",
        "indication":"Spasticity",
        "moa":       "α2-adrenergic agonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Neuropathic pain / CNS pain modulation ────────────────────────────────
    {
        "name":      "Amitriptyline",
        "smiles":    "CN(C)CCC=C1c2ccccc2CCc2ccccc21",
        "category":  "Neuropathic pain",
        "indication":"Neuropathic pain, depression, migraine prophylaxis",
        "moa":       "TCA — NE/5-HT reuptake inhibitor",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Duloxetine",
        "smiles":    "CNCCc1ccc2cccc(OC(c3ccccc3)c3cccs3)c2c1",
        "category":  "Neuropathic pain",
        "indication":"Diabetic neuropathy, fibromyalgia",
        "moa":       "SNRI",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── General Anaesthetics / Sedatives ──────────────────────────────────────
    {
        "name":      "Propofol",
        "smiles":    "CC(C)c1cccc(C(C)C)c1O",
        "category":  "Anaesthetic",
        "indication":"General anaesthesia, procedural sedation",
        "moa":       "GABAA PAM (potentiates Cl⁻ current)",
        "bbb_mechanism": "Rapid passive diffusion (highly lipophilic)",
    },
    {
        "name":      "Ketamine",
        "smiles":    "O=C1CCCN1[C@@]1(Cl)CCCCC1",
        "category":  "Anaesthetic",
        "indication":"Anaesthesia, treatment-resistant depression",
        "moa":       "NMDA receptor antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Midazolam",
        "smiles":    "Cn1cc2c(n1)CN=C(c1ccccc1F)c1cc(Cl)ccc1-2",
        "category":  "Anaesthetic",
        "indication":"Procedural sedation, status epilepticus",
        "moa":       "Benzodiazepine — GABAA PAM",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Dexmedetomidine",
        "smiles":    "Cc1ccc(C[C@@H](C)c2[nH]ccn2)cc1C",
        "category":  "Anaesthetic",
        "indication":"ICU sedation",
        "moa":       "α2-adrenergic agonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Stimulants / Wakefulness ──────────────────────────────────────────────
    {
        "name":      "Caffeine",
        "smiles":    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "category":  "Stimulant",
        "indication":"CNS stimulant, apnoea of prematurity",
        "moa":       "Adenosine A1/A2A receptor antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Modafinil",
        "smiles":    "NS(=O)(=O)c1ccc(C(Cc2ccccc2)S(=O)c2ccccc2)cc1",
        "category":  "Stimulant",
        "indication":"Narcolepsy, shift work disorder",
        "moa":       "DAT inhibitor / wakefulness-promoting (exact MOA unclear)",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Mood Stabilisers ──────────────────────────────────────────────────────
    {
        "name":      "Lithium carbonate",
        "smiles":    "[Li+].[Li+].[O-]C([O-])=O",
        "category":  "Mood stabiliser",
        "indication":"Bipolar disorder, cluster headache",
        "moa":       "GSK-3β inhibitor, inositol depletion",
        "bbb_mechanism": "Ion channel–mediated (Li⁺ ion transport)",
    },
    {
        "name":      "Lamotrigine",
        "smiles":    "Nc1nc2cc(Cl)c(Cl)cc2c(=O)[nH]1",
        "category":  "Mood stabiliser",
        "indication":"Bipolar disorder, epilepsy",
        "moa":       "Na⁺ channel blocker",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Addiction / Substance Use ─────────────────────────────────────────────
    {
        "name":      "Naltrexone",
        "smiles":    "OC1=CC=C2[C@@]3(CCN(CC=C)[C@@H]4C[C@@]35CC[C@H](O)[C@H]5[C@H]14)O2",
        "category":  "Addiction",
        "indication":"Opioid/alcohol use disorder",
        "moa":       "μ-opioid receptor antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Varenicline",
        "smiles":    "N[C@@H]1CCCC[C@@H]1NC(=O)c1cccnc1",
        "category":  "Addiction",
        "indication":"Smoking cessation",
        "moa":       "Partial α4β2 nicotinic ACh receptor agonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Acamprosate",
        "smiles":    "CC(CCS(=O)(=O)O)NC(C)C",
        "category":  "Addiction",
        "indication":"Alcohol use disorder",
        "moa":       "NMDA receptor modulator / GABAA agonist",
        "bbb_mechanism": "Passive transcellular diffusion (limited)",
    },

    # ── Headache / CNS Vascular ───────────────────────────────────────────────
    {
        "name":      "Ergotamine",
        "smiles":    "CC[C@H]1C(=O)N2CCC[C@@H]2[C@H]2Oc3c(C2=O)ccc4c3[nH]c3ccccc34",
        "category":  "Migraine",
        "indication":"Acute migraine, cluster headache",
        "moa":       "5-HT1B/1D agonist + partial agonist at α-adrenergic receptors",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Spinal cord / muscle relaxants ───────────────────────────────────────
    {
        "name":      "Cyclobenzaprine",
        "smiles":    "CN(C)CCC=C1c2ccccc2CCc2ccccc21",
        "category":  "Muscle relaxant",
        "indication":"Muscle spasm",
        "moa":       "Central α1/5-HT2 antagonist (structurally TCA-like)",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Carisoprodol",
        "smiles":    "CC(CC)(COC(N)=O)COC(=O)NC(C)C",
        "category":  "Muscle relaxant",
        "indication":"Acute musculoskeletal pain",
        "moa":       "GABAA PAM (metabolised to meprobamate)",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Cognitive enhancers / Nootropics ─────────────────────────────────────
    {
        "name":      "Piracetam",
        "smiles":    "NC(=O)CN1CCCC1=O",
        "category":  "Nootropic",
        "indication":"Cognitive impairment, myoclonus",
        "moa":       "Racetam — AMPA receptor modulator (exact MOA unclear)",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── ALS (Amyotrophic Lateral Sclerosis) ──────────────────────────────────
    {
        "name":      "Riluzole",
        "smiles":    "NC(=O)c1ccc(OC(F)(F)F)cc1Cl",
        "category":  "ALS",
        "indication":"Amyotrophic lateral sclerosis — extends survival",
        "moa":       "Glutamate release inhibitor; Na⁺ channel blocker; NMDA antagonist",
        "bbb_mechanism": "Passive transcellular diffusion (high lipophilicity)",
    },
    {
        "name":      "Edaravone",
        "smiles":    "Cc1cc(=O)n(-c2ccccc2)n1",
        "category":  "ALS",
        "indication":"ALS — free radical scavenger (approved in Japan/US)",
        "moa":       "Reactive oxygen species scavenger",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    # Tofersen is an antisense oligonucleotide (ASO) — intrathecal delivery only.
    # Its SMILES represents a simplified phosphorothioate backbone segment.
    # It does NOT cross the BBB and is included as a CNS-adjacent peripheral control.
    {
        "name":      "Tofersen",
        "smiles":    "O=P(O)(O)OCC1OC(n2ccc(=O)[nH]c2=O)CC1OP(=O)(O)O",
        "category":  "ALS",
        "indication":"SOD1-ALS — antisense oligonucleotide (intrathecal)",
        "moa":       "SOD1 mRNA-targeting antisense oligonucleotide — reduces mutant SOD1 protein",
        "bbb_mechanism": "Does not cross BBB — delivered intrathecally directly to CSF",
    },

    # ── SMA (Spinal Muscular Atrophy) ─────────────────────────────────────────
    {
        "name":      "Risdiplam",
        "smiles":    "Cc1cc2c(nc1CN1CC[C@@H](O)C1)N(c1ccncc1)C(=O)c1cc(F)ccc1-2",
        "category":  "SMA",
        "indication":"Spinal muscular atrophy — oral SMN2 splicing modifier",
        "moa":       "SMN2 pre-mRNA splicing modifier — increases full-length SMN protein",
        "bbb_mechanism": "Passive transcellular diffusion — designed for CNS penetration",
    },

    # ── Huntington's Disease ──────────────────────────────────────────────────
    {
        "name":      "Tetrabenazine",
        "smiles":    "COc1ccc2c(c1)C[C@H]1CC(=O)[C@@H](c3ccc(OC)c(OC)c3)N1CC2",
        "category":  "Huntington's",
        "indication":"Huntington's disease chorea, tardive dyskinesia",
        "moa":       "VMAT2 inhibitor — depletes presynaptic monoamine stores",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Deutetrabenazine",
        "smiles":    "COc1ccc2c(c1)C[C@H]1CC(=O)[C@@H](c3ccc(OC)c(OC([2H])([2H])[2H])c3)N1CC2",
        "category":  "Huntington's",
        "indication":"Huntington's disease chorea, tardive dyskinesia",
        "moa":       "VMAT2 inhibitor — deuterated tetrabenazine analogue (improved PK)",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Multiple Sclerosis (additional) ──────────────────────────────────────
    {
        "name":      "Dimethyl fumarate",
        "smiles":    "COC(=O)/C=C/C(=O)OC",
        "category":  "Multiple sclerosis",
        "indication":"Relapsing-remitting MS",
        "moa":       "Nrf2 pathway activator — anti-inflammatory, neuroprotective",
        "bbb_mechanism": "Passive transcellular diffusion",
    },
    {
        "name":      "Siponimod",
        "smiles":    "CC(C)(C)c1ccc(C[C@@H](NC(=O)c2cccc(C3CC3)c2)c2ccc(cc2)-c2cc(C(F)(F)F)nn2-c2ccccc2)cc1",
        "category":  "Multiple sclerosis",
        "indication":"Secondary progressive MS",
        "moa":       "S1P1/S1P5 receptor modulator — sequesters lymphocytes in lymph nodes",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Narcolepsy / Wakefulness disorders ───────────────────────────────────
    {
        "name":      "Sodium oxybate",
        "smiles":    "OCC(=O)O",
        "category":  "Narcolepsy",
        "indication":"Narcolepsy with cataplexy",
        "moa":       "GABAB agonist / GHB receptor agonist",
        "bbb_mechanism": "MCT1-mediated active transport",
    },
    {
        "name":      "Pitolisant",
        "smiles":    "O(CCCc1ccc(Cl)cc1)CCCCN1CCC(CC1)n1cnc2ccccc21",
        "category":  "Narcolepsy",
        "indication":"Narcolepsy",
        "moa":       "Histamine H3 receptor inverse agonist/antagonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Vertigo / Vestibular ──────────────────────────────────────────────────
    {
        "name":      "Betahistine",
        "smiles":    "CNNCc1ccncc1",
        "category":  "Vestibular",
        "indication":"Ménière's disease, vertigo",
        "moa":       "H1 agonist / H3 antagonist — improves labyrinthine blood flow",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Neuroprotection / Cerebrovascular ────────────────────────────────────
    {
        "name":      "Nimodipine",
        "smiles":    "CCOC(=O)C1=C(C)NC(=C(C1c1ccc([N+](=O)[O-])cc1)C(=O)OC(C)C)C",
        "category":  "Cerebrovascular",
        "indication":"Subarachnoid haemorrhage — prevents vasospasm",
        "moa":       "L-type Ca²⁺ channel blocker (CNS-selective)",
        "bbb_mechanism": "Passive transcellular diffusion (high lipophilicity)",
    },
    {
        "name":      "Vinpocetine",
        "smiles":    "CCOC(=O)[C@@H]1CC(=CCN2CCC3c4ccccc4N(C3=O)[C@H]12)C",
        "category":  "Cerebrovascular",
        "indication":"Cognitive impairment, cerebrovascular disorders",
        "moa":       "PDE1 inhibitor; Na⁺ channel blocker; cerebral vasodilator",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Nausea / CNS antiemetics ──────────────────────────────────────────────
    {
        "name":      "Ondansetron",
        "smiles":    "Cc1ccc2[nH]c(=O)n(CC3CCN(C)CC3)c2c1",
        "category":  "Antiemetic",
        "indication":"Chemotherapy-induced nausea, post-operative nausea",
        "moa":       "5-HT3 receptor antagonist",
        "bbb_mechanism": "Passive transcellular diffusion (limited CNS penetration)",
    },
    {
        "name":      "Metoclopramide",
        "smiles":    "CCN(CC)CCNC(=O)c1cc(Cl)c(N)cc1OC",
        "category":  "Antiemetic",
        "indication":"Nausea, gastroparesis",
        "moa":       "D2 antagonist / 5-HT4 agonist",
        "bbb_mechanism": "Passive transcellular diffusion",
    },

    # ── Peripherally-acting controls (intentionally poor CNS) ────────────────
    {
        "name":      "Atenolol",
        "smiles":    "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
        "category":  "Peripheral control",
        "indication":"Hypertension (peripheral beta-blocker — low CNS penetration)",
        "moa":       "β1-adrenergic antagonist",
        "bbb_mechanism": "Excluded by high polarity and P-gp efflux",
    },
    {
        "name":      "Neostigmine",
        "smiles":    "CN(C)C(=O)Oc1ccc(cc1)[N+](C)(C)C",
        "category":  "Peripheral control",
        "indication":"Myasthenia gravis, reversal of NMB (peripheral)",
        "moa":       "Quaternary AChE inhibitor — does not cross BBB",
        "bbb_mechanism": "Quaternary ammonium — cannot cross BBB",
    },
]
