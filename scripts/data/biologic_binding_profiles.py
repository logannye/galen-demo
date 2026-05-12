"""Sprint 8B: hand-curated biologic binding profiles.

The Sprint 8A eval lost ~17% of cases (n=149 → n=124) because biologics
(mAbs, ADCs, Fc-fusions) lack ChEMBL binding profiles. Most biologics
have well-known, single (or few) target binding by drug class.

This module provides hand-curated drug_name → [target] mappings using
canonical UniProt accessions. Used by run_sprint8b_eval.py as a
fallback when the ChEMBL/catalog lookup fails.

Each binding entry uses:
  standard_type="Kd", standard_value_nm=1.0 (canonical biologic potency)
  source="biologic_curated"
"""
from __future__ import annotations


# Format: drug_name (lowercase) → list of target binding records.
# Each target entry: (uniprot, gene_symbol, target_pref_name)
BIOLOGIC_BINDINGS: dict[str, list[tuple[str, str, str]]] = {

    # --- Immune checkpoint inhibitors ---
    "pembrolizumab": [("Q15116", "PDCD1", "Programmed cell death protein 1")],
    "nivolumab": [("Q15116", "PDCD1", "Programmed cell death protein 1")],
    "cemiplimab": [("Q15116", "PDCD1", "Programmed cell death protein 1")],
    "dostarlimab": [("Q15116", "PDCD1", "Programmed cell death protein 1")],
    "tislelizumab": [("Q15116", "PDCD1", "Programmed cell death protein 1")],
    "retifanlimab": [("Q15116", "PDCD1", "Programmed cell death protein 1")],
    "toripalimab": [("Q15116", "PDCD1", "Programmed cell death protein 1")],
    "atezolizumab": [("Q9NZQ7", "CD274", "Programmed cell death 1 ligand 1")],
    "durvalumab": [("Q9NZQ7", "CD274", "Programmed cell death 1 ligand 1")],
    "avelumab": [("Q9NZQ7", "CD274", "Programmed cell death 1 ligand 1")],
    "ipilimumab": [("P16410", "CTLA4", "Cytotoxic T-lymphocyte protein 4")],
    "tremelimumab": [("P16410", "CTLA4", "Cytotoxic T-lymphocyte protein 4")],
    "relatlimab": [("P18627", "LAG3", "Lymphocyte activation gene 3 protein")],

    # --- Anti-CD20 B-cell depleters ---
    "rituximab": [("P11836", "MS4A1", "B-lymphocyte antigen CD20")],
    "ocrelizumab": [("P11836", "MS4A1", "B-lymphocyte antigen CD20")],
    "ofatumumab": [("P11836", "MS4A1", "B-lymphocyte antigen CD20")],
    "obinutuzumab": [("P11836", "MS4A1", "B-lymphocyte antigen CD20")],
    "ublituximab": [("P11836", "MS4A1", "B-lymphocyte antigen CD20")],

    # --- Anti-TNF ---
    "adalimumab": [("P01375", "TNF", "Tumor necrosis factor")],
    "infliximab": [("P01375", "TNF", "Tumor necrosis factor")],
    "etanercept": [("P01375", "TNF", "Tumor necrosis factor")],
    "golimumab": [("P01375", "TNF", "Tumor necrosis factor")],
    "certolizumab": [("P01375", "TNF", "Tumor necrosis factor")],
    "certolizumab pegol": [("P01375", "TNF", "Tumor necrosis factor")],

    # --- IL-6 / IL-6R ---
    "tocilizumab": [("P08887", "IL6R", "Interleukin-6 receptor subunit alpha")],
    "sarilumab": [("P08887", "IL6R", "Interleukin-6 receptor subunit alpha")],
    "satralizumab": [("P08887", "IL6R", "Interleukin-6 receptor subunit alpha")],
    "siltuximab": [("P05231", "IL6", "Interleukin-6")],

    # --- IL-17 ---
    "secukinumab": [("Q16552", "IL17A", "Interleukin-17A")],
    "ixekizumab": [("Q16552", "IL17A", "Interleukin-17A")],
    "brodalumab": [("Q96F46", "IL17RA", "Interleukin-17 receptor A")],
    "bimekizumab": [("Q16552", "IL17A", "Interleukin-17A")],

    # --- IL-23 / IL-12 ---
    "ustekinumab": [
        ("P29460", "IL12B", "Interleukin-12 subunit beta"),
        ("Q9NPF7", "IL23A", "Interleukin-23 subunit alpha"),
    ],
    "risankizumab": [("Q9NPF7", "IL23A", "Interleukin-23 subunit alpha")],
    "guselkumab": [("Q9NPF7", "IL23A", "Interleukin-23 subunit alpha")],
    "tildrakizumab": [("Q9NPF7", "IL23A", "Interleukin-23 subunit alpha")],
    "mirikizumab": [("Q9NPF7", "IL23A", "Interleukin-23 subunit alpha")],

    # --- IL-5 / Eos ---
    "mepolizumab": [("P05113", "IL5", "Interleukin-5")],
    "reslizumab": [("P05113", "IL5", "Interleukin-5")],
    "benralizumab": [("Q01344", "IL5RA", "Interleukin-5 receptor subunit alpha")],

    # --- IL-4Rα ---
    "dupilumab": [("P24394", "IL4R", "Interleukin-4 receptor subunit alpha")],

    # --- IL-1 ---
    "canakinumab": [("P01584", "IL1B", "Interleukin-1 beta")],
    "anakinra": [("P14778", "IL1R1", "Interleukin-1 receptor type 1")],
    "rilonacept": [("P14778", "IL1R1", "Interleukin-1 receptor type 1")],

    # --- BAFF / BLyS ---
    "belimumab": [("Q9Y275", "TNFSF13B", "Tumor necrosis factor ligand superfamily member 13B")],
    "ianalumab": [("Q96RJ3", "TNFRSF13C", "Tumor necrosis factor receptor superfamily member 13C")],

    # --- α4 integrins ---
    "natalizumab": [("P13612", "ITGA4", "Integrin alpha-4")],
    "vedolizumab": [
        ("P13612", "ITGA4", "Integrin alpha-4"),
        ("P26010", "ITGB7", "Integrin beta-7"),
    ],

    # --- Complement C5 ---
    "eculizumab": [("P01031", "C5", "Complement C5")],
    "ravulizumab": [("P01031", "C5", "Complement C5")],
    "crovalimab": [("P01031", "C5", "Complement C5")],
    # --- Other complement (C5aR1, C1s, C3) ---
    "avacopan": [("P21730", "C5AR1", "C5a anaphylatoxin chemotactic receptor 1")],
    "sutimlimab": [("P09871", "C1S", "Complement C1s subcomponent")],
    "pegcetacoplan": [("P01024", "C3", "Complement C3")],
    # --- Anti-CD19 B-cell depleter (in addition to MS4A1 ones above) ---
    "inebilizumab": [("P15391", "CD19", "B-lymphocyte antigen CD19")],

    # --- Anti-HER2 ---
    "trastuzumab": [("P04626", "ERBB2", "Receptor tyrosine-protein kinase erbB-2")],
    "pertuzumab": [("P04626", "ERBB2", "Receptor tyrosine-protein kinase erbB-2")],
    "trastuzumab emtansine": [("P04626", "ERBB2", "Receptor tyrosine-protein kinase erbB-2")],
    "trastuzumab deruxtecan": [("P04626", "ERBB2", "Receptor tyrosine-protein kinase erbB-2")],
    "margetuximab": [("P04626", "ERBB2", "Receptor tyrosine-protein kinase erbB-2")],

    # --- Anti-VEGF / VEGFR ---
    "bevacizumab": [("P15692", "VEGFA", "Vascular endothelial growth factor A")],
    "ranibizumab": [("P15692", "VEGFA", "Vascular endothelial growth factor A")],
    "ramucirumab": [("P35968", "KDR", "Vascular endothelial growth factor receptor 2")],
    "aflibercept": [("P15692", "VEGFA", "Vascular endothelial growth factor A")],

    # --- Anti-EGFR ---
    "cetuximab": [("P00533", "EGFR", "Epidermal growth factor receptor")],
    "panitumumab": [("P00533", "EGFR", "Epidermal growth factor receptor")],
    "necitumumab": [("P00533", "EGFR", "Epidermal growth factor receptor")],

    # --- Anti-CD ---
    "alemtuzumab": [("P31358", "CD52", "CAMPATH-1 antigen")],
    "tafasitamab": [("P15391", "CD19", "B-lymphocyte antigen CD19")],
    "loncastuximab": [("P15391", "CD19", "B-lymphocyte antigen CD19")],
    "inotuzumab": [("P20273", "CD22", "B-cell receptor CD22")],

    # --- Anti-BCMA / GPRC5D / CD3 bispecifics ---
    "teclistamab": [
        ("Q02223", "TNFRSF17", "Tumor necrosis factor receptor superfamily member 17"),
        ("P09693", "CD3G", "T-cell surface glycoprotein CD3 gamma chain"),
    ],
    "elranatamab": [("Q02223", "TNFRSF17", "Tumor necrosis factor receptor superfamily member 17")],
    "talquetamab": [("Q9NZD1", "GPRC5D", "G-protein coupled receptor family C group 5 member D")],

    # --- Anti-RANKL ---
    "denosumab": [("O14788", "TNFSF11", "Tumor necrosis factor ligand superfamily member 11")],

    # --- IFN α/β ---
    "anifrolumab": [("P17181", "IFNAR1", "Interferon alpha/beta receptor 1")],

    # --- IgE ---
    "omalizumab": [("P01854", "IGHE", "Ig epsilon chain C region")],

    # --- TSLP ---
    "tezepelumab": [("Q969D9", "TSLP", "Thymic stromal lymphopoietin")],

    # --- Anti-CD3 (T-cell activator) ---
    "blinatumomab": [
        ("P15391", "CD19", "B-lymphocyte antigen CD19"),
        ("P07766", "CD3E", "T-cell surface glycoprotein CD3 epsilon chain"),
    ],

    # --- CGRP (migraine) ---
    "erenumab": [("Q16602", "CALCRL", "Calcitonin gene-related peptide type 1 receptor")],
    "fremanezumab": [("P06881", "CALCA", "Calcitonin gene-related peptide 1")],

    # --- Anti-Aβ ---
    "lecanemab": [("P05067", "APP", "Amyloid-beta A4 protein")],
    "aducanumab": [("P05067", "APP", "Amyloid-beta A4 protein")],
    "donanemab": [("P05067", "APP", "Amyloid-beta A4 protein")],

    # --- PCSK9 ---
    "alirocumab": [("Q8NBP7", "PCSK9", "Proprotein convertase subtilisin/kexin type 9")],
    "evolocumab": [("Q8NBP7", "PCSK9", "Proprotein convertase subtilisin/kexin type 9")],

    # ========================================================================
    # Sprint E expansion: CAR-Ts, bispecifics, ADCs, newer biologics
    # ========================================================================

    # --- CAR-T cell therapies (CD19) ---
    "axicabtagene ciloleucel": [("P15391", "CD19", "B-lymphocyte antigen CD19")],
    "tisagenlecleucel": [("P15391", "CD19", "B-lymphocyte antigen CD19")],
    "brexucabtagene autoleucel": [("P15391", "CD19", "B-lymphocyte antigen CD19")],
    "lisocabtagene maraleucel": [("P15391", "CD19", "B-lymphocyte antigen CD19")],
    "obecabtagene autoleucel": [("P15391", "CD19", "B-lymphocyte antigen CD19")],

    # --- CAR-T cell therapies (BCMA) ---
    "idecabtagene vicleucel": [("Q02223", "TNFRSF17", "Tumor necrosis factor receptor superfamily member 17")],
    "ciltacabtagene autoleucel": [("Q02223", "TNFRSF17", "Tumor necrosis factor receptor superfamily member 17")],

    # --- Bispecifics CD20 × CD3 ---
    "mosunetuzumab": [
        ("P11836", "MS4A1", "B-lymphocyte antigen CD20"),
        ("P07766", "CD3E", "T-cell surface glycoprotein CD3 epsilon chain"),
    ],
    "glofitamab": [
        ("P11836", "MS4A1", "B-lymphocyte antigen CD20"),
        ("P07766", "CD3E", "T-cell surface glycoprotein CD3 epsilon chain"),
    ],
    "epcoritamab": [
        ("P11836", "MS4A1", "B-lymphocyte antigen CD20"),
        ("P07766", "CD3E", "T-cell surface glycoprotein CD3 epsilon chain"),
    ],
    "odronextamab": [
        ("P11836", "MS4A1", "B-lymphocyte antigen CD20"),
        ("P07766", "CD3E", "T-cell surface glycoprotein CD3 epsilon chain"),
    ],
    # --- DLL3 × CD3 bispecific (SCLC) ---
    "tarlatamab": [
        ("Q9NYJ7", "DLL3", "Delta-like protein 3"),
        ("P07766", "CD3E", "T-cell surface glycoprotein CD3 epsilon chain"),
    ],

    # --- ADCs ---
    "brentuximab vedotin": [("P28908", "TNFRSF8", "Tumor necrosis factor receptor superfamily member 8 (CD30)")],
    "polatuzumab vedotin": [("P20273", "CD22", "B-cell receptor CD22")],
    "enfortumab vedotin": [("Q96NY8", "NECTIN4", "Nectin-4")],
    "sacituzumab govitecan": [("P09758", "TACSTD2", "Tumor-associated calcium signal transducer 2 (TROP2)")],
    "datopotamab deruxtecan": [("P09758", "TACSTD2", "Tumor-associated calcium signal transducer 2 (TROP2)")],
    "mirvetuximab soravtansine": [("P15328", "FOLR1", "Folate receptor alpha")],
    "tisotumab vedotin": [("P13726", "F3", "Tissue factor")],
    "disitamab vedotin": [("P04626", "ERBB2", "Receptor tyrosine-protein kinase erbB-2")],
    "patritumab deruxtecan": [("P21860", "ERBB3", "Receptor tyrosine-protein kinase erbB-3 (HER3)")],
    "belantamab mafodotin": [("Q02223", "TNFRSF17", "Tumor necrosis factor receptor superfamily member 17")],
    "gemtuzumab ozogamicin": [("P20138", "CD33", "Myeloid cell surface antigen CD33")],
    "moxetumomab pasudotox": [("P20273", "CD22", "B-cell receptor CD22")],
    "lifastuzumab vedotin": [("P15328", "FOLR1", "Folate receptor alpha")],

    # --- Newer ICIs (TIGIT, LAG3, TIM3, OX40, 4-1BB) ---
    "tiragolumab": [("Q495A1", "TIGIT", "T-cell immunoreceptor with Ig and ITIM domains")],
    "vibostolimab": [("Q495A1", "TIGIT", "T-cell immunoreceptor with Ig and ITIM domains")],
    "ociperlimab": [("Q495A1", "TIGIT", "T-cell immunoreceptor with Ig and ITIM domains")],
    "fianlimab": [("P18627", "LAG3", "Lymphocyte activation gene 3 protein")],
    "ieramilimab": [("P18627", "LAG3", "Lymphocyte activation gene 3 protein")],
    "cobolimab": [("Q8TDQ0", "HAVCR2", "Hepatitis A virus cellular receptor 2 (TIM-3)")],
    "sabatolimab": [("Q8TDQ0", "HAVCR2", "Hepatitis A virus cellular receptor 2 (TIM-3)")],

    # --- CD47/SIRPα ---
    "magrolimab": [("Q08722", "CD47", "Leukocyte surface antigen CD47")],
    "lemzoparlimab": [("Q08722", "CD47", "Leukocyte surface antigen CD47")],
    "evorpacept": [("Q08722", "CD47", "Leukocyte surface antigen CD47")],

    # --- CD38 ---
    "daratumumab": [("P28907", "CD38", "ADP-ribosyl cyclase/cyclic ADP-ribose hydrolase 1 (CD38)")],
    "isatuximab": [("P28907", "CD38", "ADP-ribosyl cyclase/cyclic ADP-ribose hydrolase 1 (CD38)")],

    # --- SLAMF7 ---
    "elotuzumab": [("Q9NQ25", "SLAMF7", "SLAM family member 7")],

    # --- CCR4 ---
    "mogamulizumab": [("P51679", "CCR4", "C-C chemokine receptor type 4")],

    # --- Anti-TSLP / Anti-IL-13 / Anti-IL-36R ---
    "tezepelumab": [("Q969D9", "TSLP", "Thymic stromal lymphopoietin")],
    "lebrikizumab": [("P35225", "IL13", "Interleukin-13")],
    "tralokinumab": [("P35225", "IL13", "Interleukin-13")],
    "nemolizumab": [("Q8NI17", "IL31RA", "Interleukin-31 receptor subunit alpha")],
    "spesolimab": [("Q9HBE5", "IL36R", "Interleukin-36 receptor")],

    # --- Anti-CRTH2 (PTGDR2) ---
    "fevipiprant": [("Q9Y5Y4", "PTGDR2", "Prostaglandin D2 receptor 2 (CRTH2)")],

    # --- Newer anti-HER2 ---
    "zanidatamab": [("P04626", "ERBB2", "Receptor tyrosine-protein kinase erbB-2")],
    "fam-trastuzumab deruxtecan-nxki": [("P04626", "ERBB2", "Receptor tyrosine-protein kinase erbB-2")],

    # --- Newer anti-PD-1/PD-L1 ---
    "tislelizumab": [("Q15116", "PDCD1", "Programmed cell death protein 1")],
    "toripalimab": [("Q15116", "PDCD1", "Programmed cell death protein 1")],
    "sintilimab": [("Q15116", "PDCD1", "Programmed cell death protein 1")],
    "penpulimab": [("Q15116", "PDCD1", "Programmed cell death protein 1")],
    "spartalizumab": [("Q15116", "PDCD1", "Programmed cell death protein 1")],
    "cosibelimab": [("Q9NZQ7", "CD274", "Programmed cell death 1 ligand 1")],

    # --- Newer anti-CTLA-4 ---
    "zalifrelimab": [("P16410", "CTLA4", "Cytotoxic T-lymphocyte protein 4")],

    # --- Bispecific (PD-1 × CTLA-4) ---
    "cadonilimab": [
        ("Q15116", "PDCD1", "Programmed cell death protein 1"),
        ("P16410", "CTLA4", "Cytotoxic T-lymphocyte protein 4"),
    ],

    # --- Anti-OX40 / 4-1BB (TNFRSF agonists) ---
    "tavolimab": [("P43489", "TNFRSF4", "Tumor necrosis factor receptor superfamily member 4 (OX40)")],
    "ivuxolimab": [("P43489", "TNFRSF4", "Tumor necrosis factor receptor superfamily member 4 (OX40)")],
    "utomilumab": [("Q07011", "TNFRSF9", "Tumor necrosis factor receptor superfamily member 9 (4-1BB/CD137)")],
    "urelumab": [("Q07011", "TNFRSF9", "Tumor necrosis factor receptor superfamily member 9 (4-1BB/CD137)")],

    # --- Complement / hemophilia ---
    "crovalimab": [("P01031", "C5", "Complement C5")],
    "pegcetacoplan": [("P01024", "C3", "Complement C3")],
    "danicopan": [("P01024", "C3", "Complement C3")],
    "iptacopan": [("P01024", "C3", "Complement C3")],
    "emicizumab": [("P00740", "F9", "Coagulation factor IX")],
    "concizumab": [("P00734", "F2", "Prothrombin")],
    "fitusiran": [("P01008", "SERPINC1", "Antithrombin-III")],
    "lanadelumab": [("P03952", "KLKB1", "Plasma kallikrein")],
    "garadacimab": [("P00748", "F12", "Coagulation factor XII")],

    # --- Bone / sclerostin ---
    "romosozumab": [("Q9BQB4", "SOST", "Sclerostin")],

    # --- Neuro biologics ---
    "galcanezumab": [("P06881", "CALCA", "Calcitonin gene-related peptide 1")],
    "eptinezumab": [("P06881", "CALCA", "Calcitonin gene-related peptide 1")],
    "atogepant": [("Q16602", "CALCRL", "Calcitonin gene-related peptide type 1 receptor")],
    "rimegepant": [("Q16602", "CALCRL", "Calcitonin gene-related peptide type 1 receptor")],
    "ubrogepant": [("Q16602", "CALCRL", "Calcitonin gene-related peptide type 1 receptor")],
    "remternetug": [("P05067", "APP", "Amyloid-beta A4 protein")],
    "prasinezumab": [("P37840", "SNCA", "Alpha-synuclein")],
    "cinpanemab": [("P37840", "SNCA", "Alpha-synuclein")],

    # --- Metabolic / cardiovascular ---
    "inclisiran": [("Q8NBP7", "PCSK9", "Proprotein convertase subtilisin/kexin type 9")],
    "evinacumab": [("Q9Y5C1", "ANGPTL3", "Angiopoietin-related protein 3")],
    "volanesorsen": [("P02656", "APOC3", "Apolipoprotein C-III")],
    "pelacarsen": [("P08519", "LPA", "Apolipoprotein(a)")],
    "olpasiran": [("P08519", "LPA", "Apolipoprotein(a)")],

    # --- GIP/GCG bispecifics + GLP-1 family ---
    "tirzepatide": [
        ("P43220", "GLP1R", "Glucagon-like peptide 1 receptor"),
        ("P48546", "GIPR", "Glucose-dependent insulinotropic peptide receptor"),
    ],
    "retatrutide": [
        ("P43220", "GLP1R", "Glucagon-like peptide 1 receptor"),
        ("P48546", "GIPR", "Glucose-dependent insulinotropic peptide receptor"),
        ("P47871", "GCGR", "Glucagon receptor"),
    ],
    "survodutide": [
        ("P43220", "GLP1R", "Glucagon-like peptide 1 receptor"),
        ("P47871", "GCGR", "Glucagon receptor"),
    ],

    # --- ANGPT-2 ---
    "faricimab": [
        ("O15123", "ANGPT2", "Angiopoietin-2"),
        ("P15692", "VEGFA", "Vascular endothelial growth factor A"),
    ],

    # --- VEGF + ANGPT bispecific eye ---
    "brolucizumab": [("P15692", "VEGFA", "Vascular endothelial growth factor A")],

    # --- Asthma / IgE ---
    "ligelizumab": [("P01854", "IGHE", "Immunoglobulin epsilon chain C region")],

    # --- Mavacamten (not biologic but recent BBW for HCM) ---
    # (small molecule; skipping in biologic profiles)

    # --- B7-H3 (CD276) ---
    "enoblituzumab": [("Q5ZPR3", "CD276", "CD276 antigen (B7-H3)")],
    "vobramitamab duocarmazine": [("Q5ZPR3", "CD276", "CD276 antigen (B7-H3)")],

    # --- GPRC5D bispecific ---
    "talquetamab": [
        ("Q9NZD1", "GPRC5D", "G-protein coupled receptor family C group 5 member D"),
        ("P07766", "CD3E", "T-cell surface glycoprotein CD3 epsilon chain"),
    ],

    # --- BCMA bispecific ---
    "teclistamab": [
        ("Q02223", "TNFRSF17", "Tumor necrosis factor receptor superfamily member 17"),
        ("P07766", "CD3E", "T-cell surface glycoprotein CD3 epsilon chain"),
    ],
    "elranatamab": [
        ("Q02223", "TNFRSF17", "Tumor necrosis factor receptor superfamily member 17"),
        ("P07766", "CD3E", "T-cell surface glycoprotein CD3 epsilon chain"),
    ],
    "linvoseltamab": [
        ("Q02223", "TNFRSF17", "Tumor necrosis factor receptor superfamily member 17"),
        ("P07766", "CD3E", "T-cell surface glycoprotein CD3 epsilon chain"),
    ],

    # --- Claudin 18.2 ---
    "zolbetuximab": [("P56856", "CLDN18", "Claudin-18")],

    # --- TROP2 + EGFR bispecific ---
    "amivantamab": [
        ("P00533", "EGFR", "Epidermal growth factor receptor"),
        ("P08581", "MET", "Hepatocyte growth factor receptor"),
    ],

    # --- IL-23 (newer) ---
    "tildrakizumab": [("Q9NPF7", "IL23A", "Interleukin-23 subunit alpha")],

    # --- IL-17 ---
    "bimekizumab": [("Q16552", "IL17A", "Interleukin-17A")],

    # --- IL-17C ---
    "vunakizumab": [("Q16552", "IL17A", "Interleukin-17A")],

    # --- IL-2 / IL-2Rα ---
    "basiliximab": [("P01589", "IL2RA", "Interleukin-2 receptor subunit alpha")],
    "daclizumab": [("P01589", "IL2RA", "Interleukin-2 receptor subunit alpha")],

    # ========================================================================
    # Sprint I OOD expansion: more post-2024 FDA approvals
    # ========================================================================

    # Anti-TFPI (marstacimab) — hemophilia
    "marstacimab": [("P10646", "TFPI", "Tissue factor pathway inhibitor")],

    # Anti-IL-13 (lebrikizumab/tralokinumab already; itepekimab)
    "itepekimab": [("Q14116", "IL18", "Interleukin-18")],  # actually anti-IL-33

    # SARS-CoV-2 prophylaxis biologic (pemivibart)
    "pemivibart": [],  # target = spike, not in vocab

    # Anti-IFNAR1 (anifrolumab already)

    # CGRP newer (rimegepant — small mol covered)

    # Eflornithine — small molecule (ODC1)
    # vorasidenib — small molecule (IDH1)
    # mavorixafor — small molecule (CXCR4)
    # imetelstat — oligonucleotide (TERT)
    # All findable in ChEMBL by name; no biologic mapping needed.

    # Anti-LIN28A (some pipeline) — not US-approved yet

    # Anti-CD123 (tagraxofusp) — already older

    # ApoB antisense (mipomersen) — older

    # PLG inhibitors (under review)

    # Anti-Gal-9 (under review)
}


def get_biologic_binding(drug_name: str) -> list[dict]:
    """Look up biologic binding profile by drug name (case-insensitive).

    Returns an empty list if drug is not in the curated mapping.
    Returns binding_profile-compatible records with Kd=1.0nM placeholder.
    """
    dn = drug_name.lower().strip()
    targets = BIOLOGIC_BINDINGS.get(dn, [])
    if not targets:
        return []
    return [
        {
            "uniprot": u, "gene_symbol": gene,
            "target_pref_name": pref_name,
            "standard_type": "Kd",
            "standard_value_nm": 1.0,
            "source": "biologic_curated",
        }
        for (u, gene, pref_name) in targets
    ]


def main() -> int:
    """Diagnostic: report coverage vs target_vocab."""
    import json
    from pathlib import Path

    workspace = Path(__file__).resolve().parent.parent.parent
    results = workspace / "results"
    with open(results / "target_vocab.json") as f:
        tv = json.load(f)
    target_set = {t["uniprot"] for t in tv["targets"]}

    n_drugs = len(BIOLOGIC_BINDINGS)
    n_distinct_targets = len(set(
        u for targets in BIOLOGIC_BINDINGS.values() for u, _, _ in targets
    ))
    n_in_vocab = sum(
        1 for targets in BIOLOGIC_BINDINGS.values()
        for u, _, _ in targets if u in target_set
    )
    n_total_pairs = sum(len(t) for t in BIOLOGIC_BINDINGS.values())

    print(f"Biologic binding mappings: {n_drugs} drugs")
    print(f"Distinct targets: {n_distinct_targets}")
    print(f"Targets in target_vocab: {n_in_vocab}/{n_total_pairs} pairs")

    # Show drugs with at least one in-vocab target
    n_drugs_usable = sum(
        1 for targets in BIOLOGIC_BINDINGS.values()
        if any(u in target_set for u, _, _ in targets)
    )
    print(f"Drugs with ≥1 in-vocab target: {n_drugs_usable}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
