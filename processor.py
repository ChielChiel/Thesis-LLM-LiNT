from multiprocessing import Pool
import tqdm

from lint_ii import ReadabilityAnalysis
import pandas as pd
import numpy as np
import Rewrite as rw
from pathlib import Path
import json
from datetime import datetime

n_pools = 1
output_dir = Path("./output/meta_prompt_no_lint_formula")
n_texts = 40

base_prompt = """
De leesbaarheidsscore wordt als volgt berekend: 100 - (
        - 4.21
        + (17.28 * woordfrequentie)
        - (1.62 * syntactische afhankelijkheidslengte)
        - (2.54 * inhoudswoorden per bijzin)
        + (16.00 * aandeel concrete zelfstandige naamwoorden)
    )
Ik wil een tekst met een maximale leesbaarheidsscore van 34.
"""

standard_prompt = f"""
Jij gaat mij helpen om een tekst te herschrijven die slecht leesbaar is in een eenvoudigere leesbare tekst.
{base_prompt}
De informatie van de originele tekst moet zoveel als mogelijk behouden blijven in de herschreven, leesbaardere tekst.
Je krijgt een tekst die je moet herschrijven. Je hoeft geen vraag te beantwoorden, alleen de tekst te herschrijven.
"""

roleplay_prompt = f"""
Je speelt de rol van een goede schrijver. Je werkt bij Gebruiker Centraal die overheidsbrieven versimpelt. Je bent goed in het herschrijven van moeilijk leesbare teksten zodat
deze eenvoudiger leesbaar zijn. Je kan altijd teksten met een hoge leesbaarheidsscore omschrijven naar teksten met een leesbaarheidsscore onder de 34.
{base_prompt}
Hoe lager de score, hoe beter. Jij hebt als taak gekregen om de teksten die je krijgt te versimpelen aan de hand van de leesbaarheidsscore.
Het doel is om laaggeletterde mensen de moeilijk leesbare teksten te laten begrijpen. Bij het herschrijven behoud je de inhoud van de tekst en doe je je wijzigingen op zinsniveau.
Je voegt dus geen dingen toe en de structuur van de tekst blijft behouden.
"""

meta_prompt = f"""
{base_prompt}
Het probleem met de volgende tekst is dat deze een leesbaarheidsscore van <doc_score> heeft. De leesbaarheidsscore van de zinnen zijn <sent_scores>.
Herschrijf de tekst zodat het een lagere leesbaarheidsscore heeft. Los het probleem op door het op te splitsen in kleinere stapjes. Geef de nieuwe tekst in het gegeven format
"""

cot_prompt = f"""
Doel: Het doel is om de leesbaarheidsscore van een tekst zo laag mogelijk te maken. Hoe lager de score, hoe beter.
{base_prompt}
De huidige leesbaarheidsscore van de tekst is <doc_score>; het doel is 34 of lager. De huidige scores per zin zijn <sent_scores>.

Methode:
- Denk eerst na voor je de tekst herschrijft. Beredeneer waarom je een aanpassing maakt, verwijs naar de specifieke teksten.
- Bepaal eerst het thema van de tekst. Bepaal op basis daarvan of moeilijke woorden bepalend zijn voor de betekenis van de tekst.
- Herschrijf geen zinnen met een leesbaarheidsscore lager dan 34, behoud deze wel.
- Lange woorden zijn waarschijnlijk moeilijk. Vervang alleen deze door eenvoudigere alternatieven.
- Bijvoeglijknaamwoorden mogen vereenvoudigd of verwijderd worden als ze niet essentieel zijn.
- Soms horen twee woorden in de zin bij elkaar, maar staan ze ver van elkaar af. Die onderlinge afstand noemen we de afhankelijkheidslengte. Herschrijf zodat deze niet langer is dan 8 woorden. 
- Maak van deelzinnen aparte zinnen. Gebruik signaalwoorden om de verbinding te houden met originele hoofdzin.
- Eigennamen en entiteiten moeten behouden blijven.
- Houd de betekenis en de boodschap van de zinnen en de tekst hetzelfde.
- Voeg geen informatie toe.
- Geef de nieuwe tekst in het gegeven format.
"""

meta_prompt_no_metric = f"""
{base_prompt}
Het probleem met de volgende tekst is dat deze een te hoge leesbaarheidsscore heeft.
Herschrijf de tekst zodat het een lagere leesbaarheidsscore heeft. Los het probleem op door het op te splitsen in kleinere stapjes. Geef de nieuwe tekst in het gegeven format
"""

meta_prompt_no_lint_formula = f"""
Het probleem met de volgende tekst is dat deze een te hoge leesbaarheidsscore heeft.
Herschrijf de tekst zodat het een lagere leesbaarheidsscore heeft. Los het probleem op door het op te splitsen in kleinere stapjes. Geef de nieuwe tekst in het gegeven format
"""

chatgpt_41 = rw.RE_OpenAi(
    model="gpt-4.1-mini",
    name = "ChatGPT 4.1 mini",
    is_local=False,
    is_open_source=False,
    is_large=True,
    is_dutch=True
)

chatgpt_51 = rw.RE_OpenAi(
    model="gpt-5-mini",
    name = "GPT 5.1 mini",
    is_local=False,
    is_open_source=False,
    is_large=True,
    is_dutch=True
)

gemini = rw.RE_Gemini(
    model="gemini-3-flash-preview",
    name="Gemini",
    is_local=False,
    is_open_source=False,
    is_large=True,
    is_dutch=True
)

mistral = rw.RE_Mistral(
    model="ministral-14b-latest", 
    name="Mistral", 
    timeout_ms=240 * 1000,
    is_local=True,
    is_open_source=True,
    is_large=False,
    is_dutch=True
    )


all_prompts = [standard_prompt, roleplay_prompt, meta_prompt, cot_prompt]
print("Run meta_prompt_no_metric prompt")

evaluation = None


# for index, row in texts.iterrows():

def run(row):
    # print(index, row['id'], row['text'])
    text_id = row['id']
    PROMPT = meta_prompt_no_lint_formula
    all_pipe = rw.LintPipe(instruction_en=PROMPT, instruction_nl=PROMPT, user_prompt=row['text'], prompt_id=text_id)
    # all_pipe.add_engine(chatgpt_41)
    # all_pipe.add_engine(chatgpt_51)
    all_pipe.add_engine(gemini)
    # all_pipe.add_engine(mistral)
    # print("Prompting engines...")
    all_pipe.prompt_engines()
    # print("evaluation")
    evals = all_pipe.eval_response()
    evals["created"] = str(datetime.now())

    output_fp = output_dir / f"{text_id}.redone.json"
    output_fp.parent.mkdir(exist_ok=True, parents=True)

    with open(output_fp, "w") as f:
        output = evals.iloc[:1].to_dict(orient="records")[0]
        f.write(json.dumps(output, indent=4))

    return output

    # if index == 0:
    #     evaluation = evals
    # else:
    #     evaluation = pd.concat([evaluation, evals])
    
    # if index % 20 == 0 or index == texts.shape[0] - 1:
    #     print("saving progress till", index)
    #     evaluation.to_parquet(f"./output/df_stand_{index}.parquet.gzip", compression="gzip")


if __name__ == '__main__':
    texts = pd.read_csv("./prep_texts.csv")
    # texts.head(5)

    if isinstance(n_texts, type(None)):
        n_texts = len(texts)

    # Missing values for meta prompt per model. Lists contain indices of texts dataframe.
    missing_values_meta_no_lint = {'ChatGPT 4.1 mini/gpt-4.1-mini': [],
        'GPT 5.1 mini/gpt-5-mini': [38],
        'Gemini/gemini-3-flash-preview': [28, 39],
        'Mistral/ministral-14b-latest': []}
    
    # Only run model for these indices.
    # Now only running GEMINI <-- redo if needed.
    texts_input = texts.loc[missing_values_meta_no_lint['Gemini/gemini-3-flash-preview']].to_dict(orient="records")

    with Pool(processes=n_pools) as pool:
        results = list(tqdm.tqdm(iterable=pool.imap(func=run, iterable=texts_input), total=len(texts_input)))

    print("Hoera! Gelukt! Denk ik...")

    results_df = pd.DataFrame(data=results)

    output_fp = output_dir / "gemini-meta_prompt_no_lint_formula.redone.parquet"
    results_df.to_parquet(path=output_fp, compression="gzip")

    print(f"Opgeslagen als: `{output_fp}`")