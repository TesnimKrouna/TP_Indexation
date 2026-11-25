import math, re, csv
from pathlib import Path
from collections import Counter, defaultdict

# Reprise du pretraitement du TP3 fourni
STOPWORDS = {"a","an","and","are","as","at","be","been","being","but","by","for","from","had","has","have","he","her","here","hers","him","his","i","if","in","into","is","it","its","me","more","most","my","no","not","of","on","or","our","ours","she","so","such","than","that","the","their","theirs","them","then","there","these","they","this","those","to","too","under","up","us","was","we","were","what","when","where","which","who","why","will","with","you","your"}
BASE = Path(__file__).resolve().parent
K1_DEFAULT, B_DEFAULT = 1.2, 0.75

def tokeniser(texte):
    return re.findall(r"[a-z]+", texte.lower())

def stemmer_simple(mot):
    if len(mot)>4 and mot.endswith("ies"): return mot[:-3]+"y"
    if len(mot)>5 and mot.endswith(("ches","shes","xes","zes","sses","oes")): return mot[:-2]
    if len(mot)>3 and mot.endswith("s"): return mot[:-1]
    return mot

def preprocess(texte, use_stemming=False, remove_stopwords=True):
    mots=tokeniser(texte)
    if remove_stopwords: mots=[m for m in mots if m not in STOPWORDS]
    if use_stemming: mots=[stemmer_simple(m) for m in mots]
    return mots

def charger_documents():
    docs={}
    with open(BASE/"documents.tsv",encoding="utf-8") as f:
        for no,line in enumerate(f,1):
            line=line.strip()
            if not line: continue
            parts=line.split("\t",1)
            if len(parts)!=2: print(f"Ligne {no} ignoree"); continue
            if parts[0] in docs: print("ID duplique:",parts[0]); continue
            docs[parts[0]]=parts[1]
    return docs

def construire_index_enrichi(documents,use_stemming=False):
    index=defaultdict(lambda:{"df":0,"postings":{}})
    for did,txt in documents.items():
        for terme,tf in Counter(preprocess(txt,use_stemming)).items():
            index[terme]["postings"][did]={"tf":tf}
            index[terme]["df"]+=1
    return dict(index)

# Exercice 1
def construire_stats(documents,use_stemming=False):
    longueurs={d:len(preprocess(t,use_stemming)) for d,t in documents.items()}
    avgdl=sum(longueurs.values())/len(longueurs) if longueurs else 0
    return longueurs,avgdl

def afficher_stats(longueurs,avgdl):
    print("\nEX1 - 5 plus courts:",sorted(longueurs.items(),key=lambda x:(x[1],x[0]))[:5])
    print("EX1 - 5 plus longs:",sorted(longueurs.items(),key=lambda x:(-x[1],x[0]))[:5])
    print(f"N={len(longueurs)}, avgdl={avgdl:.4f}")
    print("|D| se calcule apres preprocess(), car BM25 normalise la longueur des termes indexes.")

# Exercice 2
def calculer_idf(t,index,N):
    if t not in index or N<=0:return 0.0
    return math.log10(N/index[t]["df"])
def idf_bm25(t,index,N):
    if t not in index or N<=0:return 0.0
    df=index[t]["df"]
    return math.log(1+(N-df+0.5)/(df+0.5))
def idf_rsj(t,index,N):
    if t not in index or N<=0:return 0.0
    df=index[t]["df"]
    return math.log((N-df+0.5)/(df+0.5))
def afficher_idf(index,N):
    print("\nEX2 - TERME | DF | IDF TFIDF | IDF BM25 | IDF RSJ")
    for t in sorted(index):
        print(f"{t:<18} {index[t]['df']:>3} {calculer_idf(t,index,N):>9.4f} {idf_bm25(t,index,N):>9.4f} {idf_rsj(t,index,N):>9.4f}")
    print("RSJ devient negatif quand df > N/2 ; Lucene reste positif.")

# Exercice 3
def bm25_search(query,index,doc_len,avgdl,N,k1=1.2,b=.75,k=5,details=False):
    terms=list(dict.fromkeys(preprocess(query)))
    scores=defaultdict(float); contrib=defaultdict(dict); visited=set()
    for t in terms:
        if t not in index: continue
        idf=idf_bm25(t,index,N)
        for did,info in index[t]["postings"].items():
            visited.add(did); tf=info["tf"]
            K=k1*(1-b+b*doc_len[did]/avgdl) if avgdl else 0
            c=idf*(tf*(k1+1))/(tf+K)
            scores[did]+=c; contrib[did][t]=c
    ranked=sorted(scores.items(),key=lambda x:(-x[1],x[0]))
    if k is not None: ranked=ranked[:k]
    if details:return ranked,len(visited),contrib
    return ranked

def poids_tfidf(t,did,index,N):
    info=index.get(t,{}).get("postings",{}).get(did)
    return (1+math.log10(info["tf"]))*calculer_idf(t,index,N) if info else 0.0
def cosine_similarity(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    return sum(x*y for x,y in zip(a,b))/(na*nb) if na and nb else 0.0
def tfidf_search(query,docs,index,vocab,N,k=None):
    qtf=Counter(preprocess(query))
    qv=[(1+math.log10(qtf[t]))*calculer_idf(t,index,N) if qtf[t] else 0 for t in vocab]
    out=[]
    for did in docs:
        dv=[poids_tfidf(t,did,index,N) for t in vocab]
        score=cosine_similarity(qv,dv)
        if score>0:out.append((did,score))
    out.sort(key=lambda x:(-x[1],x[0]))
    return out[:k] if k is not None else out
def recherche_booleenne_and(query,index):
    terms=list(dict.fromkeys(preprocess(query)))
    if not terms or any(t not in index for t in terms):return []
    return sorted(set.intersection(*[set(index[t]["postings"]) for t in terms]))

# Exercice 4
def test_mini_corpus():
    mini={"D1":"information retrieval systems","D2":"information retrieval models vector","D3":"boolean retrieval model","D4":"vector space model information"}
    idx=construire_index_enrichi(mini)
    lens={d:len(tokeniser(t)) for d,t in mini.items()}
    avg=sum(lens.values())/4
    expected={"D3":1.1150,"D4":.9919,"D1":.7576,"D2":.6740}
    result=bm25_search("information retrieval model",idx,lens,avg,4,k=4)
    print("\nEX4 - Mini corpus, avgdl=",avg)
    for d,s in result:
        print(d,round(s,4),"attendu",expected[d])
        assert abs(s-expected[d])<1e-3
    assert [d for d,s in result]==["D3","D4","D1","D2"]
    K=1.2*(.25+.75*lens["D3"]/avg)
    print("D3: K=",round(K,4))
    for t in ("retrieval","model"):
        tf=idx[t]["postings"]["D3"]["tf"]
        print(t,"IDF",round(idf_bm25(t,idx,4),4),"contribution",round(idf_bm25(t,idx,4)*tf*2.2/(tf+K),4))
    # Verification externe facultative avec rank_bm25 (non utilise pour le calcul manuel).
    try:
        from rank_bm25 import BM25Okapi
        corpus = [preprocess(t, remove_stopwords=False) for t in mini.values()]
        controle = BM25Okapi(corpus, k1=1.2, b=0.75)
        print("Controle rank_bm25 (scores externes, IDF different) :")
        for i, did in enumerate(mini):
            print(did, round(controle.get_scores(preprocess("information retrieval model", remove_stopwords=False))[i], 4))
        print("Des ecarts sont normaux : BM25Okapi utilise une variante RSJ avec plancher epsilon.")
    except ImportError:
        print("rank_bm25 non installe : controle externe ignore (pip install rank_bm25).")
    print("TEST EX4 OK")

# Exercice 5
def experience_saturation():
    print("\nEX5 - SATURATION TF")
    ks=[0,.5,1.2,2,10]
    for k1 in ks:
        vals=[tf*(k1+1)/(tf+k1) for tf in (1,3,10)]
        print("k1=",k1,"TFsat(tf=1,3,10)=",[round(x,4) for x in vals])
    try:
        import matplotlib.pyplot as plt
        x=list(range(31)); plt.figure()
        for k1 in ks:
            y=[0 if tf==0 else tf*(k1+1)/(tf+k1) for tf in x]
            plt.plot(x,y,label=f"k1={k1}")
            plt.axhline(k1+1,linestyle="--",linewidth=.7)
        plt.plot(x,x,label="TF brut")
        plt.xlabel("TF");plt.ylabel("TF sature");plt.title("Saturation du TF")
        plt.legend();plt.grid();plt.tight_layout();plt.savefig(BASE/"ex5_saturation_tf.png",dpi=150);plt.close()
        print("Graphique ex5_saturation_tf.png cree")
    except ImportError: print("Installer matplotlib pour le graphique.")
    print("k1=0 : facteur=1 pour tf>0. k1 tres grand : TFsat tend vers TF.")

# Exercice 6
def experience_longueur(docs):
    print("\nEX6 - NORMALISATION DE LONGUEUR")
    local={d:docs[d] for d in ("D29","D30")}
    lens,avg=construire_stats(local);idx=construire_index_enrichi(local)
    bs=[0,.25,.5,.75,1];ratios=[]
    print("Longueurs:",lens,"avgdl:",avg)
    for b in bs:
        vals=[]
        for d in ("D29","D30"):
            K=1.2*(1-b+b*lens[d]/avg);tf=idx["blockchain"]["postings"][d]["tf"]
            comp=tf*2.2/(tf+K);score=idf_bm25("blockchain",idx,2)*comp
            vals.append((K,comp,score))
        rho=vals[0][2]/vals[1][2];ratios.append(rho)
        print(f"b={b}: court K/TFsat/score={vals[0]}, long={vals[1]}, rho={rho:.4f}")
    try:
        import matplotlib.pyplot as plt
        plt.figure();plt.plot(bs,ratios,marker="o");plt.xlabel("b");plt.ylabel("score court / score long")
        plt.title("Normalisation de longueur");plt.grid();plt.tight_layout()
        plt.savefig(BASE/"ex6_normalisation_longueur.png",dpi=150);plt.close()
    except ImportError: print("Installer matplotlib pour le graphique.")
    print("b=0 rend BM25 insensible a la longueur.")

# TSV
def charger_requetes():
    with open(BASE/"queries.tsv",encoding="utf-8") as f:
        return {r["QueryID"]:r["Texte"] for r in csv.DictReader(f,delimiter="\t")}
def charger_qrels():
    q=defaultdict(dict)
    with open(BASE/"qrels.tsv",encoding="utf-8") as f:
        for r in csv.DictReader(f,delimiter="\t"):q[r["QueryID"]][r["DocID"]]=int(r["Grade"])
    return dict(q)

# Exercice 9 metrics
def precision_at_k(ranking,qrels_q,k):
    return sum(qrels_q.get(d,0)>=1 for d in ranking[:k])/k if k>0 else 0
def recall_at_k(ranking,qrels_q,k):
    total=sum(g>=1 for g in qrels_q.values())
    return sum(qrels_q.get(d,0)>=1 for d in ranking[:k])/total if total else 0
def average_precision(ranking,qrels_q):
    total=sum(g>=1 for g in qrels_q.values())
    if not total:return 0
    hits=0;s=0
    for i,d in enumerate(ranking,1):
        if qrels_q.get(d,0)>=1:hits+=1;s+=hits/i
    return s/total
def dcg_at_k(ranking,qrels_q,k):
    return sum(qrels_q.get(d,0)/math.log2(i+2) for i,d in enumerate(ranking[:k]))
def ndcg_at_k(ranking,qrels_q,k):
    ideal=sorted(qrels_q.values(),reverse=True)[:k]
    idcg=sum(g/math.log2(i+2) for i,g in enumerate(ideal))
    return dcg_at_k(ranking,qrels_q,k)/idcg if idcg else 0
def test_metriques():
    ranking=["A","B","C","D","E"];q={"A":2,"B":0,"C":1,"D":2,"F":1}
    assert abs(precision_at_k(ranking,q,5)-.6)<1e-4
    assert abs(recall_at_k(ranking,q,5)-.75)<1e-4
    assert abs(average_precision(ranking,q)-.6041667)<1e-4
    assert abs(ndcg_at_k(ranking,q,5)-.8017)<1e-3
    print("\nTEST EX9 METRIQUES OK")

def evaluer(docs,index,vocab,lens,avg,queries,qrels,model="bm25",k1=1.2,b=.75):
    values=[]
    for qid,query in queries.items():
        if model=="bm25":rank=[d for d,s in bm25_search(query,index,lens,avg,len(docs),k1,b,None)]
        elif model=="tfidf":rank=[d for d,s in tfidf_search(query,docs,index,vocab,len(docs),None)]
        else:rank=recherche_booleenne_and(query,index)
        rel=qrels.get(qid,{})
        values.append((precision_at_k(rank,rel,5),recall_at_k(rank,rel,5),average_precision(rank,rel),ndcg_at_k(rank,rel,5)))
    return tuple(sum(v[i] for v in values)/len(values) for i in range(4)) if values else (0,0,0,0)

def comparaison(docs,index,vocab,lens,avg,queries,qrels):
    print("\nEX7 - TOP 5 TF-IDF / BM25")
    for qid,q in queries.items():
        a=tfidf_search(q,docs,index,vocab,len(docs),5);b=bm25_search(q,index,lens,avg,len(docs),k=5)
        print("\n",qid,q,"\nTF-IDF:",[(d,round(s,4)) for d,s in a],"\nBM25:",[(d,round(s,4)) for d,s in b])
        print("Top5 communs:",len(set(d for d,s in a)&set(d for d,s in b)))
        _, visites, _ = bm25_search(q,index,lens,avg,len(docs),k1=1.2,b=.75,k=None,details=True)
        print(f"Documents visites : BM25={visites}/{len(docs)} ; TF-IDF cosinus={len(docs)}/{len(docs)}")
        try:
            from scipy.stats import spearmanr
            # rang des documents non retournes place apres les resultats
            def ranks(res):
                r={d:i+1 for i,(d,s) in enumerate(res)}
                return [r.get(d,len(r)+1) for d in docs]
            full_a=tfidf_search(q,docs,index,vocab,len(docs),None)
            full_b=bm25_search(q,index,lens,avg,len(docs),k=None)
            print("Spearman:",round(spearmanr(ranks(full_a),ranks(full_b)).statistic,4))
        except ImportError: print("scipy absent : Spearman non calcule")

    print("\nEX9 - EVALUATION")
    print("Modele              P@5     R@5     MAP     nDCG@5")
    for m,name in (("and","Boolean AND"),("tfidf","TF-IDF"),("bm25","BM25")):
        p,r,ap,n=evaluer(docs,index,vocab,lens,avg,queries,qrels,m)
        print(f"{name:<18}{p:>7.4f}{r:>8.4f}{ap:>8.4f}{n:>10.4f}")
    print("Boolean AND : ordre alphabetique des DocID, sans score de ranking.")

# Exercice 10
def optimisation(docs,index,vocab,lens,avg,queries,qrels):
    ids=list(queries);train={q:queries[q] for q in ids[:5]};test={q:queries[q] for q in ids[5:]}
    qt={q:qrels.get(q,{}) for q in train};qv={q:qrels.get(q,{}) for q in test}
    ks=[.5,.8,1.2,1.6,2.0];bs=[0,.25,.5,.75,1];grid=[];best=(-1,None,None)
    for k1 in ks:
        row=[]
        for b in bs:
            score=evaluer(docs,index,vocab,lens,avg,train,qt,"bm25",k1,b)[2];row.append(score)
            if score>best[0]:best=(score,k1,b)
        grid.append(row)
    print("\nEX10 - MAP GRID (rows k1, columns b)")
    print("       ",bs)
    for k1,row in zip(ks,grid):print(k1,[round(x,4) for x in row])
    def m(q,r,k,b):return evaluer(docs,index,vocab,lens,avg,q,r,"bm25",k,b)[2]
    print("Default train/test:",m(train,qt,1.2,.75),m(test,qv,1.2,.75))
    print("Best train pair:",best[1:], "MAP train/test:",m(train,qt,best[1],best[2]),m(test,qv,best[1],best[2]))
    try:
        import matplotlib.pyplot as plt
        plt.figure();plt.imshow(grid,origin="lower",aspect="auto");plt.xticks(range(5),bs);plt.yticks(range(5),ks)
        plt.xlabel("b");plt.ylabel("k1");plt.title("MAP apprentissage");plt.colorbar(label="MAP")
        plt.tight_layout();plt.savefig(BASE/"ex10_heatmap_map.png",dpi=150);plt.close()
    except ImportError:print("Installer matplotlib pour la heatmap.")
    print("Optimiser sur les memes requetes que le test provoque un risque de sur-apprentissage.")

# Exercice 12
def moteur_interactif(docs,index,vocab,lens,avg,queries,qrels):
    model="bm25";k=5;k1=1.2;b=.75
    print("\nFIGL SEARCH ENGINE V3 | :model 1/2/3 | :k 3/5/10 | :params 1.2 0.75 | :eval | :quit")
    while True:
        q=input("Query > ").strip()
        if q==":quit":break
        if q.startswith(":model"):
            try:model={1:"tfidf",2:"bm25",3:"and"}[int(q.split()[1])];print("Modele:",model)
            except:print("Choisir 1, 2 ou 3")
            continue
        if q.startswith(":k"):
            try:k=int(q.split()[1]);assert k in (3,5,10)
            except:print("Utiliser :k 3, :k 5 ou :k 10")
            continue
        if q.startswith(":params"):
            try:
                _,a,bb=q.split();k1=float(a);b=float(bb);assert k1>=0 and 0<=b<=1
            except:print("Exemple :params 1.2 0.75")
            continue
        if q==":eval":
            print("P@5, R@5, MAP, nDCG@5:",evaluer(docs,index,vocab,lens,avg,queries,qrels,model,k1,b));continue
        if model=="bm25":
            res,n,details=bm25_search(q,index,lens,avg,len(docs),k1,b,k,True)
            for i,(d,s) in enumerate(res,1):
                print(i,d,round(s,5),docs[d],details[d])
            print("Documents visites:",n,"/",len(docs))
        elif model=="tfidf":
            for i,(d,s) in enumerate(tfidf_search(q,docs,index,vocab,len(docs),k),1):print(i,d,round(s,5),docs[d])
        else:print([(d,docs[d]) for d in recherche_booleenne_and(q,index)])

def main():
    docs=charger_documents();N=len(docs);index=construire_index_enrichi(docs);vocab=sorted(index)
    lens,avg=construire_stats(docs)
    print("TP4 BM25 - Corpus:",N,"documents | vocabulaire:",len(vocab))
    afficher_stats(lens,avg);afficher_idf(index,N)
    test_mini_corpus();test_metriques();experience_saturation();experience_longueur(docs)
    queries=charger_requetes();qrels=charger_qrels()
    comparaison(docs,index,vocab,lens,avg,queries,qrels)
    optimisation(docs,index,vocab,lens,avg,queries,qrels)
    if input("\nDemarrer moteur interactif ? (o/n) ").lower()=="o":
        moteur_interactif(docs,index,vocab,lens,avg,queries,qrels)

if __name__=="__main__":main()
