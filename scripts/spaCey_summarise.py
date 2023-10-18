import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

def summarize(text, per):
    nlp = spacy.load('en_core_web_sm')
    doc= nlp(text)
    tokens=[token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summary=''.join(final_summary)
    return summary


def main():
    
    text = "Greta Thunberg has been charged with a public order offence after she was arrested while taking part in a protest against a conference in London described as “the Oscars of oil" \
    "The Swedish environmental campaigner, 20, was one of 29 arrested during a protest trying to stop delegates entering the Energy Intelligence Forum at the InterContinental Park Lane in Mayfair, central London."\
"Videos showed her being placed in the back of a police van after joining crowds blocking hotel entrances and chanting “oily money out”. Police officers had jostled with protesters to create a channel for delegates to enter."\
"Thunberg had earlier given a speech to journalists denouncing the conference, a meeting of energy executives and politicians, saying: “We have no other option but to put our bodies outside this conference and to physically disrupt [it]."\
"According to the Metropolitan police, she was charged with “failing to comply with a condition imposed under section 14 of the Public Order Act”. Police had demanded protesters move from the road on to the pavement."\
"Twenty-six others were also charged with various alleged offences relating to the protest."\
"Details of Thunberg’s charge came as Just Stop Oil said its cofounders, Indigo Rumbelow and Roger Hallam, were arrested on Wednesday morning following dawn raids at their homes."\
"The climate campaign said police had forced entry to their homes, searched their belongings and confiscated papers. But it added that supporters would continue with plans to march every day in London from 30 October. “We will not be intimidated by our criminal government,” a spokesperson said."\
"Not content with cheering on war crimes in Gaza, by maxing out our oil and gas reserves they are complicit in the greatest crime in human history … no one has ever voted for this, there has never been a democratic mandate to destroy the habitable world."\
"Meanwhile, protests against the Energy Intelligence Forum continued on Wednesday, with protesters occupying the offices of 10 insurers, demanding they rule out insuring the proposed West Cumbria coalmine and the East Africa crude oil pipeline."\
"Protesters had gathered outside the headquarters of Standard Bank in the City of London financial district before marching to each the site of each occupation."\
"Claude Fourcroy, a spokesperson for Money Rebellion, said: “We are calling on all the banks and insurers behind the West Cumbria mine and East Africa crude oil pipeline to cut their ties now. Both of these projects will fuel climate breakdown. Lloyd’s of London and the insurers in its market sit at the centre of a web of climate wreckers in the City of London."


    print(summarize(text, 0.25))

if __name__ == "__main__":
    main()

