# Elon Tweet Analysis

## Panoramica
Questo repository contiene due script Python pensati per analizzare la frequenza dei post pubblici di Elon Musk e per stimare quante nuove pubblicazioni potrebbero apparire in una finestra temporale specifica. I dati provengono da esportazioni CSV dell account X/Twitter e includono identificativo del post, testo e timestamp (senza anno). Gli script ricostruiscono i timestamp completi, generano statistiche giornaliere e salvano grafici e riassunti tabellari.

## Struttura dei dati
- `elonmusk (2).csv`: esportazione manuale con tre colonne (id, body, created_at) prive di quoting standard.
- `elon_posts.csv` / `all_musk_posts.csv`: eventuali esportazioni alternative con schema standard.
- `elonmusk-Elon_Musk___tweets_November_25___December_2__2025_-daily-metrics.csv`: file con conteggi giornalieri manuali (vedi sezione dedicata) per coprire giorni successivi all'ultimo export grezzo.
- `horizon_summary.csv`: tabella prodotta da `horizon_stats.py` con metriche aggregate per piu' orizzonti temporali.
- `daily_counts_*.png`: grafici creati da `analysis.py` per ogni finestra considerata.

## Script principali
### `analysis.py`
1. Ricerca automaticamente il primo CSV disponibile fra i candidati e armonizza i timestamp in UTC.
2. Filtra i post applicando euristiche sui flag `isReply`, `isRetweet` e `isQuote` quando presenti.
3. Costruisce serie giornaliere per finestre mobili (1/3/6 mesi e YTD), disegna i grafici e stima un modello Negative Binomial per trend + giorno della settimana.
4. Simula un milione di scenari per la finestra target (`WINDOW_START`-`WINDOW_END`) e stampa
   - media, mediana e intervallo al 90%
   - probabilita' che il totale cada in intervalli di 20 unita'.

### `horizon_stats.py`
1. Legge la versione manuale del CSV (necessaria perche' contiene post piu' recenti) usando il parser personalizzato.
2. Ricostruisce i timestamp completi aggiungendo l anno 2025 e convertendo in timezone `America/New_York`.
3. Calcola, per ogni finestra richiesta, conteggi totali, giorni di attivita', medie/mediane e giorno con massimo volume.
4. Salva il risultato in `horizon_summary.csv` e stampa la tabella.

## Requisiti
- Python 3.11+
- Dipendenze: `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`
- Ambiente virtuale suggerito (e.g. `python -m venv .venv && source .venv/bin/activate`).

## Come eseguire
```bash
pip install -r requirements.txt  # oppure installa manualmente i pacchetti richiesti
python analysis.py --window-start 2025-11-28 --window-end 2025-12-09 --time-remaining 7D --known-total 0
python horizon_stats.py
```

`analysis.py` ora espone tutti i parametri principali via CLI:
- `--window-start` / `--window-end`: estremi (UTC) della finestra da simulare.
- `--time-remaining`: durata che separa "adesso" dalla fine finestra (Timedelta tipo `3D12H`).
- `--known-total`: post gia' contati nella finestra (sottratti ai forecast).
- `--current-timestamp`: in alternativa a `--time-remaining`, passa l'istante UTC dell'ultimo post censito.

Entrambi gli script scrivono i risultati nella radice del progetto. `analysis.py` mostra stampe verbose: per ridurre l output e' possibile commentare alcune `print` oppure reindirizzare lo stdout su file.

## Aggiornare i dati
1. Scarica un nuovo CSV con le tre colonne richieste e rinominalo in `elonmusk (2).csv`.
2. Verifica che i timestamp riportino mese, giorno e orario nel fuso di New York.
3. Lancia gli script per rigenerare grafici e tabelle.

### Aggiornare manualmente i giorni piu' recenti
Quando l export completo non e' ancora disponibile, utilizza `elonmusk-Elon_Musk___tweets_November_25___December_2__2025_-daily-metrics.csv` come diario giornaliero:

1. Ogni riga segue il formato `YYYY-MM-DD,POSTS,cumulative_posts`. Il terzo campo e' opzionale: se vuoi puoi aggiornarlo con la somma progressiva, ma il software usa solo la colonna `posts`.
2. Aggiungi (o modifica) una riga per ogni data oltre l ultimo giorno contenuto in `elonmusk (2).csv`. `analysis.py` ignora automaticamente le date gia' coperte dal dataset grezzo, quindi non rischi doppi conteggi.
3. Salva il file in CSV (testo semplice, separatore `,`). Alla successiva esecuzione, lo script generera' tweet sintetici per quei giorni cosi' da includerli nelle medie giornaliere e nelle simulazioni.

In questo modo ti basta appendere due numeri per giorno: non servono ID dei singoli post. Quando l export completo sara' disponibile potrai sostituire `elonmusk (2).csv` e svuotare (o aggiornare) il file manuale.

## Risoluzione problemi
- **File mancante**: se nessun CSV viene trovato, assicurati che almeno uno dei percorsi in `DATA_CANDIDATES` esista.
- **Timestamp non parsabili**: in presenza di stringhe malformate il parser solleva un `ValueError`; correggi le righe incriminate o rimuovile.
- **Dipendenze mancanti**: installa `statsmodels` dal PyPI ufficiale (`pip install statsmodels`).

## Prossimi passi possibili
- Aggiungere il supporto a nuovi anni rilevando dinamicamente l anno corretto dal file CSV.
- Salvare automaticamente i risultati delle simulazioni in `CSV` per analisi successive.
- Integrare un notebook riassuntivo per condividere grafici e insight in formato interattivo.
