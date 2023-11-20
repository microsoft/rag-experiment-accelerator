from rag_experiment_accelerator.evaluation.eval import bleu
from rag_experiment_accelerator.evaluation.eval import answer_relevance
from rag_experiment_accelerator.evaluation.eval import context_precision


def test_bleu():
    predictions = [
        "Transformers Transformers are fast plus efficient",
        "Good Morning",
        "I am waiting for new Transformers",
    ]
    references = [
        [
            "HuggingFace Transformers are quick, efficient and awesome",
            "Transformers are awesome because they are fast to execute",
        ],
        ["Good Morning Transformers", "Morning Transformers"],
        [
            "People are eagerly waiting for new Transformer models",
            "People are very excited about new Transformers",
        ],
    ]
    score = bleu(predictions, references)
    assert round(score) == 50


# TODO: Mock out actual LLM call

# For testing scores, since it's coming from an LLM and will be different every time
# Idea: Maybe simplify the question/answer/context so relevance score can be rounded consistently?
# def test_answer_relevance():
#     question = "What is the credit challenge posed by the push towards alternative-fuel vehicles for Japan Inc.?"
#     answer = "The push towards alternative-fuel vehicles poses a credit challenge for multiple sectors in Japan, including Japanese auto manufacturers and associated industries, which will make sizable upfront investments in alternative-fuel vehicle technologies while bearing risks around the scale and speed of this vehicle take-up. The risks of miscalculating this transition are substantial.",
#     score = answer_relevance(question, answer)
#     assert score == 0.963114833440214
    

# # Same comments as above
# def test_context_precision():
#     question = "What is the credit challenge posed by the push towards alternative-fuel vehicles for Japan Inc.?"
#     context = "CORPORATES\nSECTOR IN-DEPTH\n9 April 2018\nTABLE OF CONTENTS\nAlternative-fuel vehicles will have\nsignificant effects on key Japanese\nindustries 2\nAuto manufacturers \u2013 credit negative 4\nAuto-parts suppliers \u2013 mixed credit\nimpact 5\nElectronics \u2013 credit positive 6\nSteel \u2013 credit negative 7\nChemicals \u2013 credit positive 7\nRefining \u2013 credit negative 7\nRegional and local governments \u2013\ncredit negative 8\nElectric utilities \u2013 credit positive 8\nMoody\u2019s related publications 9\nAnalyst Contacts\nMotoki Yanase +81.3.5408.4154\nVP-Sr Credit Officer Moody\u2019s Japan K.K.\nmotoki.yanase@moodys.com\nMariko Semetko +81.3.5408.4209\nVP-Sr Credit Officer Moody\u2019s Japan K.K.\nmariko.semetko@moodys.com\nMasako Kuwahara +81.3.5408.4155\nVP-Senior Analyst Moody\u2019s Japan K.K.\nmasako.kuwahara@moodys.com\nTakashi Akimoto +81.3.5408.4208\nAVP-Analyst Moody\u2019s Japan K.K.\ntakashi.akimoto@moodys.com\nMihoko Manabe, CFA +81.3.5408.4033\nAssociate Managing\nDirectorMoody\u2019s Japan K.K.\nmihoko.manabe@moodys.com\nBrian Cahill +61.2.9270.8105\nMD-Asia Pac Corp &\nInfra Fin\nbrian.cahill@moodys.comCross-Sector\nPush for alternative-fuel vehicles presents\nchallenges for Japan Inc.\n\u00bbThe push toward alternative-fuel vehicles poses a credit challenge for multiple\nsectors in Japan.  Over the next decade, Japanese auto manufacturers and associated\nindustries, which we refer to as Japan Inc., will make sizable upfront investments in\nalternative-fuel vehicle technologies while bearing risks around the scale and speed of this\nvehicle take-up. The risks of miscalculating this transition are substantial.\n\u00bbElectrification will have widespread effects on key industries.  The credit impact\non Japan Inc. is weighted to the negative because of the challenges facing the large\nauto sector, and the direct impact on sectors such as steel and refining. Lower gasoline\nconsumption will also reduce a meaningful source of government tax revenue, which\nfunds road construction and public works programs."

#     score = context_precision(question, context)
#     assert score == 1.0