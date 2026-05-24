from lint_ii import ReadabilityAnalysis
import pandas as pd
import numpy as np
import json
from enum import Enum
from typing import List

from anthropic import Anthropic
from openai import OpenAI
from google import genai
from google.genai import types
from mistralai.client import Mistral
import lmstudio as lms
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv(".env", override=True)


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import tqdm
import outlines
import os
os.environ['HF_TOKEN'] = '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'

class RewriteEngine:
     """
     Engine that rewrites a text using a specific LLM
     
     
     """
     def __init__(self, 
                  model:str, 
                  name:str, 
                  is_dutch:bool = False, 
                  is_local:bool = False, 
                  is_open_source:bool = False, 
                  is_large:bool = False):
          
          self.model = model
          self.name = name
          self.is_dutch = is_dutch
          self.is_local = is_local
          self.is_open_source = is_open_source
          self.is_large = is_large
          pass
     
     def set_instruction(self, instruction:str):
          self.instruction = instruction
     
     def prompt_model(self, user_prompt:str):
          pass
     
# Response schema
# class RewriteChange(BaseModel):
#     original_part_of_text: str
#     rewritten_part_of_text: str
#     reason_for_change: str
class RewriteCategory(str, Enum):
    word_frequency = "verhoog woordfrequentie"
    syntactic_dependency_length = "verlaag afhankelijkheidslengte"
    content_words_per_clause = "verlaag contentwoorden per deelzin"
    proportion_concrete_nouns = "verhoog de verhouding van concrete zelfstandignaamwoorden"


class RewriteChange(BaseModel):
    change_id: str = Field(description="The id of the individual rewrite change.")
    reason_for_change: str = Field(description="Explanation for why the model made this change")
    start_word: int = Field(description="Inclusive start word index in orginal text")
    end_word: int = Field(description="Inclusive end word index in orginal text")
    original_text: str
    new_text: str
    rewrite_category: RewriteCategory = Field(description="Classification of the rewrite type.")

class RewriteSentence(BaseModel):
    original_sentence: str
    rewritten_sentences: str
    changes_in_sentence: List[RewriteChange] = Field(description="Every single change gets it's own RewriteChange object.")

class RewrittenText(BaseModel):
    text: List[RewriteSentence]
    text_genre: str


# class RewriteCategory(str, Enum):
#     word_frequency = "verhoog woordfrequentie"
#     syntactic_dependency_length = "verlaag afhankelijkheidslengte"
#     content_words_per_clause = "verlaag contentwoorden per deelzin"
#     proportion_concrete_nouns = "verhoog de verhouding van concrete zelfstandignaamwoorden"

# class RewriteChange(BaseModel):
#     change_id: str 
#     start_word: int
#     end_word: int
#     original_text: str
#     new_text: str
#     reason_for_change: str
#     rewrite_category: RewriteCategory

# class RewriteSentence(BaseModel):
#     original_sentence: str
#     rewritten_sentences: str
#     changes_in_sentence: List[RewriteChange]

# class RewrittenText(BaseModel):
#     text: List[RewriteSentence]
#     text_genre: str

class RE_Mistral(RewriteEngine):

    def __init__(self, model, name, is_dutch = False, is_local = False, is_open_source = False, is_large = False, timeout_ms=None):
        super().__init__(model, name, is_dutch, is_local, is_open_source, is_large)
        self.client = Mistral(
            api_key=os.environ.get("MISTRAL_API_KEY"),
            timeout_ms=timeout_ms,
            )
        # print("key:", os.environ.get("MISTRAL_API_KEY"))

    def set_instruction(self, instruction):
        return super().set_instruction(instruction)
    
    def prompt_model(self, user_prompt):
        response = self.client.chat.parse(
          model=self.model,
          messages=[
              {
                  "role": "system",
                  "content": self.instruction
              },
              {
                  "role": "user",
                  "content": f"{user_prompt}"
              },
          ],
          response_format=RewrittenText,
          max_tokens=32000,
          top_p = 1,
        )
        return response.choices[0].message.parsed


class RE_Local(RewriteEngine):

  def __init__(self, model, name, is_dutch = False, is_local = False, is_open_source = False, is_large = False):
    super().__init__(model, name, is_dutch, is_local, is_open_source, is_large)
    # print("hic")
    self.client = outlines.from_transformers(
      AutoModelForCausalLM.from_pretrained(model, device_map="auto"),
      AutoTokenizer.from_pretrained(model)
    )
        
  def set_instruction(self, instruction):
    return super().set_instruction(instruction)
    
  def prompt_model(self, user_prompt):
    response = self.client(f"{self.instruction}\n{user_prompt}", RewrittenText)
    return response


# LMStudio hosts and manages all my local models. It optimizes gpu allocation so i don't have to.
class RE_LMStudio(RewriteEngine):

    def __init__(self, model, name, is_dutch = False, is_local = False, is_open_source = False, is_large = False):
        super().__init__(model, name, is_dutch, is_local, is_open_source, is_large)
        print("hic sunt 2.")
        self.client = lms.llm(self.model) # 'fietje-2-instruct'

    def set_instruction(self, instruction):
        return super().set_instruction(instruction)
    
    def prompt_model(self, user_prompt):
        chat = lms.Chat(self.instruction)
        chat.add_user_message(user_prompt)
        
        response = self.client.respond(
            history=chat,
            response_format=RewrittenText,
            config={
                #"temperature": 1.0,
                # "maxTokens": 50,
            },
        )

        returnObject = None
        try:
            returnObject = RewrittenText.model_validate(response.parsed)
        except:
            print("Transforming to RewrittenText failed. Using default json schema instead.")
            returnObject = response.parsed

        return returnObject
    
class RE_Claude(RewriteEngine):

    def __init__(self, model, name, is_dutch = False, is_local = False, is_open_source = False, is_large = False):
        super().__init__(model, name, is_dutch, is_local, is_open_source, is_large)
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def set_instruction(self, instruction):
        return super().set_instruction(instruction)
    
    def prompt_model(self, user_prompt):
        response = self.client.messages.parse(
            model=self.model, # "claude-opus-4-7"
            output_format=RewrittenText,
            system=self.instruction,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"{user_prompt}",
                }
            ],
        )
        return response.parsed_output
    
class RE_OpenAi(RewriteEngine):

    def __init__(self, model, name, is_dutch = False, is_local = False, is_open_source = False, is_large = False):
        super().__init__(model, name, is_dutch, is_local, is_open_source, is_large)
        self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    def set_instruction(self, instruction):
        return super().set_instruction(instruction)

    def prompt_model(self, user_prompt) -> str:
        response = self.client.responses.parse(
            model=self.model, # "gpt-4.1-nano"
            instructions=self.instruction,
            text_format = RewrittenText,
            input=[
                {
                    "role": "user",
                    "content": f"{user_prompt}",
                }
            ],
            reasoning={},
            tools=[],
            max_output_tokens=32768,
            top_p=1,
            store=True,
            include=[]
            )
        # print(f"tokens: {response.input_tokens}")
        return response.output_parsed
    
class RE_Gemini(RewriteEngine):


    def __init__(self, model, name, is_dutch = False, is_local = False, is_open_source = False, is_large = False):
        super().__init__(model, name, is_dutch, is_local, is_open_source, is_large)
        self.client = genai.Client()

    def set_instruction(self, instruction):
        return super().set_instruction(instruction)
    
    def prompt_model(self, user_prompt):
        response = self.client.models.generate_content(
            model=self.model, # "gemini-3-flash-preview" 
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="medium"),
                system_instruction=self.instruction,
                response_mime_type="application/json",
                response_schema=RewrittenText
            ),
            contents=f"{user_prompt}"
        )

        rewritten = RewrittenText.model_validate_json(response.text)
        return rewritten
    
class LintPipe:

    def __init__(self, instruction_en:str, instruction_nl:str, user_prompt:str, prompt_id:str):
        self.user_prompt = user_prompt
        self.prompt_id = prompt_id
        self.analysis = self.get_lint_analysis(self.user_prompt)
        # print(self.analysis)

        self.instruction_en = self.format_instruction(instruction_en)
        self.instruction_nl = self.format_instruction(instruction_nl, is_dutch = True)
        
        self.rewriteEngines = []
        self.prompt_responses = []
        return

    def format_instruction(self, instruction:str, is_dutch:bool = False):
        # Determine document and sentence scores:
        sent_scores = [int(sent_analysis['score']) for sent_analysis in self.analysis['sentence_stats'] if sent_analysis['score'] is not None]
        doc_score = str(int(self.analysis['document_stats']['document_lint_score']))
        
        if is_dutch:
            sent_scores = "{} en {}".format(", ".join(map(str, sent_scores[:-1])),  str(object=sent_scores[-1]))
        else:
            sent_scores = "{} and {}".format(", ".join(map(str, sent_scores[:-1])),  str(sent_scores[-1]))

        instruction = instruction.replace('<doc_score>', doc_score)
        instruction = instruction.replace('<sent_scores>', sent_scores)

        return instruction


    def add_engine(self, engine:RewriteEngine):
        # TODO: Retrieve all top_n uncommon 

        if engine.is_dutch:
            engine.set_instruction(self.instruction_nl)
        else:
            engine.set_instruction(self.instruction_en)
        self.rewriteEngines.append(engine)

    def list_engines(self):
        # print("Rewrite engines:")
        for engine in self.rewriteEngines:
            # print(f"NAME: {engine.name}, MODEL: '{engine.model}', ?NL {engine.is_dutch}, ?LOCAL: {engine.is_local}, ?OS {engine.is_open_source}, ?LARGE {engine.is_large}")
            pass

    def prompt_engines(self):
        # TODO: Prompt engines moet doen:
        # 1. Return een clean object:
        #     - Model id
        #     - Lint scores per zinsblok.
        #     - Herschreven tekst

        # - Dit object moet ook een methode hebben: extract_text en get_overall_lint_score()
        self.prompt_responses = []
        results = []
        if [] == self.rewriteEngines:
            print("No Rewrite Engines added. Use `add_engine` to do so.")
            return

        for engine in self.rewriteEngines:
            # print(f"Rewriting using {engine.name} - Model: {engine.model}")
            engine_id = f'{engine.name}/{engine.model}'
            try:
                engine_result = engine.prompt_model(self.user_prompt)
                # print(engine_result)
                if type(engine_result) == RewrittenText:
                    # print("is rewritten object")                    
                    engine_result = engine_result.model_dump()
                full_text = " ".join([rs['rewritten_sentences'] for rs in engine_result['text']])
                engine_result['full_text'] = full_text
                engine_result['prompt_id'] = self.prompt_id
                engine_result['engine'] = engine_id
                results.append(engine_result)
            except Exception as e:
                print(f"Something went wrong with {engine_id}, text: {self.prompt_id} skipping...")
                print(e)
                faulty_engine = dict()
                faulty_engine['text'] = []
                faulty_engine['text_genre'] = 'NA'
                faulty_engine['full_text'] = ''
                engine_result['prompt_id'] = self.prompt_id
                faulty_engine['engine'] = engine_id


                results.append(faulty_engine)
        # print("Returning")
        self.prompt_responses = results.copy()
        return results

    def get_lint_analysis(self, user_prompt):
        analysis = ReadabilityAnalysis.from_text(user_prompt)
        return analysis.get_detailed_analysis()
    
    def eval_response(self):
        org_sent_scores = map(int, [sent_analysis['score'] for sent_analysis in self.analysis['sentence_stats'] if sent_analysis['score'] is not None])
        org_sent_scores = ", ".join(map(str, org_sent_scores))
        org_doc_score = str(int(self.analysis['document_stats']['document_lint_score']))
        
        response_object = {"prompt_id": self.prompt_id, "Org - text": self.user_prompt, "Org - lint": org_doc_score, "Org - sent": org_sent_scores}
        
        for resp in self.prompt_responses:
            # print("bezig met", resp['engine'])
            # print(resp['text'])
            analysis = ReadabilityAnalysis.from_text(resp['full_text'])
            doc_score = sent_scores = ""
            if analysis.lint.score == None:
                doc_score = sent_scores = np.nan
            else: 
                doc_score = str(int(analysis.lint.score))
                sent_scores = map(int, analysis.lint_scores_per_sentence)
                sent_scores = ", ".join(map(str, sent_scores))
            
            tr = {f"{resp['engine']} - text": resp['full_text'], f"{resp['engine']} - lint" : doc_score, f"{resp['engine']} - sent": sent_scores, f"{resp['engine']} - chngs": resp}
            # print(tr)
            response_object.update(tr)

        return pd.DataFrame([response_object])

        
