"""A bank of short factual questions with checkable answers.

Each item: {id, category, question, answers}. `answers` lists accepted strings
(surface aliases); a generation is scored correct if any appears as a normalised
token-substring (see common.scoring.is_correct).

The bank is deliberately tilted toward *moderately obscure* facts — the regime
where a 1B model tends to hallucinate while a 3B model still answers — so that
the selection step can harvest a useful set of "1B-wrong / 3B-right" divergence
cases, plus a set of "both-right" easy controls.
"""

PROMPTS = [
    # ── world capitals (less-famous) ─────────────────────────────────────────
    {"id": "cap_au", "category": "capital", "question": "What is the capital city of Australia?", "answers": ["Canberra"]},
    {"id": "cap_ca", "category": "capital", "question": "What is the capital city of Canada?", "answers": ["Ottawa"]},
    {"id": "cap_tr", "category": "capital", "question": "What is the capital city of Turkey?", "answers": ["Ankara"]},
    {"id": "cap_kz", "category": "capital", "question": "What is the capital city of Kazakhstan?", "answers": ["Astana", "Nur-Sultan", "Nur Sultan"]},
    {"id": "cap_mm", "category": "capital", "question": "What is the capital city of Myanmar?", "answers": ["Naypyidaw", "Naypyitaw", "Nay Pyi Taw"]},
    {"id": "cap_bz", "category": "capital", "question": "What is the capital city of Brazil?", "answers": ["Brasilia", "Brasília"]},
    {"id": "cap_ng", "category": "capital", "question": "What is the capital city of Nigeria?", "answers": ["Abuja"]},
    {"id": "cap_ch", "category": "capital", "question": "What is the capital city of Switzerland?", "answers": ["Bern", "Berne"]},
    {"id": "cap_nz", "category": "capital", "question": "What is the capital city of New Zealand?", "answers": ["Wellington"]},
    {"id": "cap_za", "category": "capital", "question": "What is the executive capital of South Africa?", "answers": ["Pretoria"]},
    {"id": "cap_bt", "category": "capital", "question": "What is the capital city of Bhutan?", "answers": ["Thimphu"]},
    {"id": "cap_mn", "category": "capital", "question": "What is the capital city of Mongolia?", "answers": ["Ulaanbaatar", "Ulan Bator"]},
    {"id": "cap_lk", "category": "capital", "question": "What is the commercial capital and largest city of Sri Lanka?", "answers": ["Colombo"]},
    {"id": "cap_ma", "category": "capital", "question": "What is the capital city of Morocco?", "answers": ["Rabat"]},
    {"id": "cap_ec", "category": "capital", "question": "What is the capital city of Ecuador?", "answers": ["Quito"]},

    # ── US state capitals (the tricky ones) ──────────────────────────────────
    {"id": "us_ny", "category": "us_capital", "question": "What is the capital of the U.S. state of New York?", "answers": ["Albany"]},
    {"id": "us_ca", "category": "us_capital", "question": "What is the capital of the U.S. state of California?", "answers": ["Sacramento"]},
    {"id": "us_il", "category": "us_capital", "question": "What is the capital of the U.S. state of Illinois?", "answers": ["Springfield"]},
    {"id": "us_wa", "category": "us_capital", "question": "What is the capital of the U.S. state of Washington?", "answers": ["Olympia"]},
    {"id": "us_pa", "category": "us_capital", "question": "What is the capital of the U.S. state of Pennsylvania?", "answers": ["Harrisburg"]},
    {"id": "us_mo", "category": "us_capital", "question": "What is the capital of the U.S. state of Missouri?", "answers": ["Jefferson City"]},
    {"id": "us_or", "category": "us_capital", "question": "What is the capital of the U.S. state of Oregon?", "answers": ["Salem"]},

    # ── science / constants ──────────────────────────────────────────────────
    {"id": "sci_au", "category": "science", "question": "What is the chemical symbol for gold?", "answers": ["Au"]},
    {"id": "sci_w", "category": "science", "question": "What is the chemical symbol for tungsten?", "answers": ["W"]},
    {"id": "sci_pb", "category": "science", "question": "What is the chemical symbol for lead?", "answers": ["Pb"]},
    {"id": "sci_k", "category": "science", "question": "What is the chemical symbol for potassium?", "answers": ["K"]},
    {"id": "sci_na", "category": "science", "question": "What is the chemical symbol for sodium?", "answers": ["Na"]},
    {"id": "sci_bone", "category": "science", "question": "What is the smallest bone in the human body?", "answers": ["stapes", "stirrup"]},
    {"id": "sci_planet", "category": "science", "question": "Which planet in our solar system has the most moons as of 2024?", "answers": ["Saturn"]},
    {"id": "sci_speed", "category": "science", "question": "What is the speed of light in vacuum in kilometers per second, rounded to the nearest thousand?", "answers": ["300000", "300,000", "299792", "299,792"]},
    {"id": "sci_atomic", "category": "science", "question": "What is the atomic number of carbon?", "answers": ["6", "six"]},
    {"id": "sci_blood", "category": "science", "question": "What protein in red blood cells carries oxygen?", "answers": ["hemoglobin", "haemoglobin"]},
    {"id": "sci_dna", "category": "science", "question": "Which scientist is credited, with Francis Crick, for the 1953 double-helix model of DNA?", "answers": ["Watson", "James Watson"]},
    {"id": "sci_element79", "category": "science", "question": "Which chemical element has atomic number 79?", "answers": ["gold"]},
    {"id": "sci_hardest", "category": "science", "question": "What is the hardest known naturally occurring material?", "answers": ["diamond"]},

    # ── history / dates ──────────────────────────────────────────────────────
    {"id": "his_ww2", "category": "history", "question": "In what year did World War II end?", "answers": ["1945"]},
    {"id": "his_berlin", "category": "history", "question": "In what year did the Berlin Wall fall?", "answers": ["1989"]},
    {"id": "his_moon", "category": "history", "question": "In what year did humans first land on the Moon?", "answers": ["1969"]},
    {"id": "his_magna", "category": "history", "question": "In what year was the Magna Carta sealed?", "answers": ["1215"]},
    {"id": "his_french", "category": "history", "question": "In what year did the French Revolution begin?", "answers": ["1789"]},
    {"id": "his_columbus", "category": "history", "question": "In what year did Columbus first reach the Americas?", "answers": ["1492"]},
    {"id": "his_titanic", "category": "history", "question": "In what year did the Titanic sink?", "answers": ["1912"]},
    {"id": "his_president1", "category": "history", "question": "Who was the first President of the United States?", "answers": ["Washington", "George Washington"]},
    {"id": "his_president16", "category": "history", "question": "Who was the 16th President of the United States?", "answers": ["Lincoln", "Abraham Lincoln"]},
    {"id": "his_pharaoh", "category": "history", "question": "Which ancient Egyptian queen was the last active ruler of the Ptolemaic Kingdom?", "answers": ["Cleopatra"]},
    {"id": "his_indep", "category": "history", "question": "In what year did India gain independence from British rule?", "answers": ["1947"]},

    # ── geography / superlatives ─────────────────────────────────────────────
    {"id": "geo_river", "category": "geography", "question": "What is the longest river in the world?", "answers": ["Nile", "Amazon"]},
    {"id": "geo_mountain", "category": "geography", "question": "What is the tallest mountain above sea level on Earth?", "answers": ["Everest", "Mount Everest"]},
    {"id": "geo_desert", "category": "geography", "question": "What is the largest hot desert in the world?", "answers": ["Sahara"]},
    {"id": "geo_ocean", "category": "geography", "question": "What is the largest ocean on Earth?", "answers": ["Pacific"]},
    {"id": "geo_lake", "category": "geography", "question": "What is the largest freshwater lake by surface area in the world?", "answers": ["Superior", "Lake Superior"]},
    {"id": "geo_country", "category": "geography", "question": "What is the largest country in the world by land area?", "answers": ["Russia"]},
    {"id": "geo_smallest", "category": "geography", "question": "What is the smallest country in the world by area?", "answers": ["Vatican", "Vatican City"]},
    {"id": "geo_volcano", "category": "geography", "question": "What is the largest active volcano on Earth by volume?", "answers": ["Mauna Loa"]},
    {"id": "geo_strait", "category": "geography", "question": "Which strait separates Spain from Morocco?", "answers": ["Gibraltar", "Strait of Gibraltar"]},

    # ── literature / culture ─────────────────────────────────────────────────
    {"id": "lit_romeo", "category": "literature", "question": "Who wrote the play 'Romeo and Juliet'?", "answers": ["Shakespeare", "William Shakespeare"]},
    {"id": "lit_1984", "category": "literature", "question": "Who wrote the novel '1984'?", "answers": ["Orwell", "George Orwell"]},
    {"id": "lit_pride", "category": "literature", "question": "Who wrote the novel 'Pride and Prejudice'?", "answers": ["Austen", "Jane Austen"]},
    {"id": "lit_warpeace", "category": "literature", "question": "Who wrote the novel 'War and Peace'?", "answers": ["Tolstoy", "Leo Tolstoy"]},
    {"id": "lit_odyssey", "category": "literature", "question": "Who is the ancient Greek poet traditionally credited as author of the Odyssey?", "answers": ["Homer"]},
    {"id": "lit_mona", "category": "literature", "question": "Who painted the Mona Lisa?", "answers": ["da Vinci", "Leonardo", "Leonardo da Vinci"]},
    {"id": "lit_starry", "category": "literature", "question": "Who painted 'The Starry Night'?", "answers": ["van Gogh", "Vincent van Gogh"]},
    {"id": "lit_guernica", "category": "literature", "question": "Who painted 'Guernica'?", "answers": ["Picasso", "Pablo Picasso"]},

    # ── sports / awards ──────────────────────────────────────────────────────
    {"id": "spo_wc2018", "category": "sports", "question": "Which country won the 2018 FIFA World Cup?", "answers": ["France"]},
    {"id": "spo_wc2014", "category": "sports", "question": "Which country won the 2014 FIFA World Cup?", "answers": ["Germany"]},
    {"id": "spo_olymp2016", "category": "sports", "question": "Which city hosted the 2016 Summer Olympics?", "answers": ["Rio", "Rio de Janeiro"]},
    {"id": "spo_olymp2012", "category": "sports", "question": "Which city hosted the 2012 Summer Olympics?", "answers": ["London"]},
    {"id": "spo_marathon", "category": "sports", "question": "How many kilometers long is a marathon, rounded to the nearest whole number?", "answers": ["42"]},

    # ── language / misc ──────────────────────────────────────────────────────
    {"id": "mis_currency_jp", "category": "misc", "question": "What is the official currency of Japan?", "answers": ["yen"]},
    {"id": "mis_currency_in", "category": "misc", "question": "What is the official currency of India?", "answers": ["rupee", "rupees"]},
    {"id": "mis_currency_pl", "category": "misc", "question": "What is the official currency of Poland?", "answers": ["zloty", "złoty"]},
    {"id": "mis_currency_se", "category": "misc", "question": "What is the official currency of Sweden?", "answers": ["krona", "kronor"]},
    {"id": "mis_language_br", "category": "misc", "question": "What is the official language of Brazil?", "answers": ["Portuguese"]},
    {"id": "mis_language_at", "category": "misc", "question": "What is the official language of Austria?", "answers": ["German"]},
    {"id": "mis_continents", "category": "misc", "question": "How many continents are there on Earth, by the common seven-continent model?", "answers": ["7", "seven"]},
    {"id": "mis_planets", "category": "misc", "question": "How many planets are in our solar system as of 2024?", "answers": ["8", "eight"]},
    {"id": "mis_keys", "category": "misc", "question": "How many keys are on a standard full-size piano?", "answers": ["88"]},
    {"id": "mis_chess", "category": "misc", "question": "How many squares are on a standard chessboard?", "answers": ["64"]},
    {"id": "mis_bones", "category": "misc", "question": "How many bones are in the adult human body?", "answers": ["206"]},
    {"id": "mis_amendments", "category": "misc", "question": "How many amendments are in the United States Constitution as of 2024?", "answers": ["27", "twenty-seven", "twenty seven"]},

    # ── harder batch: more national capitals (mid-tier, 1B failure-prone) ─────
    {"id": "cap_vn", "category": "capital", "question": "What is the capital city of Vietnam?", "answers": ["Hanoi"]},
    {"id": "cap_id", "category": "capital", "question": "What is the capital city of Indonesia?", "answers": ["Jakarta"]},
    {"id": "cap_sa", "category": "capital", "question": "What is the capital city of Saudi Arabia?", "answers": ["Riyadh"]},
    {"id": "cap_no", "category": "capital", "question": "What is the capital city of Norway?", "answers": ["Oslo"]},
    {"id": "cap_fi", "category": "capital", "question": "What is the capital city of Finland?", "answers": ["Helsinki"]},
    {"id": "cap_hr", "category": "capital", "question": "What is the capital city of Croatia?", "answers": ["Zagreb"]},
    {"id": "cap_rs", "category": "capital", "question": "What is the capital city of Serbia?", "answers": ["Belgrade"]},
    {"id": "cap_ro", "category": "capital", "question": "What is the capital city of Romania?", "answers": ["Bucharest"]},
    {"id": "cap_bg", "category": "capital", "question": "What is the capital city of Bulgaria?", "answers": ["Sofia"]},
    {"id": "cap_pe", "category": "capital", "question": "What is the capital city of Peru?", "answers": ["Lima"]},
    {"id": "cap_cl", "category": "capital", "question": "What is the capital city of Chile?", "answers": ["Santiago"]},
    {"id": "cap_co", "category": "capital", "question": "What is the capital city of Colombia?", "answers": ["Bogota", "Bogotá"]},
    {"id": "cap_et", "category": "capital", "question": "What is the capital city of Ethiopia?", "answers": ["Addis Ababa"]},
    {"id": "cap_gh", "category": "capital", "question": "What is the capital city of Ghana?", "answers": ["Accra"]},
    {"id": "cap_si", "category": "capital", "question": "What is the capital city of Slovenia?", "answers": ["Ljubljana"]},
    {"id": "cap_sk", "category": "capital", "question": "What is the capital city of Slovakia?", "answers": ["Bratislava"]},

    # ── harder batch: more US state capitals ─────────────────────────────────
    {"id": "us_ky", "category": "us_capital", "question": "What is the capital of the U.S. state of Kentucky?", "answers": ["Frankfort"]},
    {"id": "us_me", "category": "us_capital", "question": "What is the capital of the U.S. state of Maine?", "answers": ["Augusta"]},
    {"id": "us_sd", "category": "us_capital", "question": "What is the capital of the U.S. state of South Dakota?", "answers": ["Pierre"]},
    {"id": "us_vt", "category": "us_capital", "question": "What is the capital of the U.S. state of Vermont?", "answers": ["Montpelier"]},
    {"id": "us_nv", "category": "us_capital", "question": "What is the capital of the U.S. state of Nevada?", "answers": ["Carson City"]},
    {"id": "us_ak", "category": "us_capital", "question": "What is the capital of the U.S. state of Alaska?", "answers": ["Juneau"]},
    {"id": "us_nd", "category": "us_capital", "question": "What is the capital of the U.S. state of North Dakota?", "answers": ["Bismarck"]},

    # ── harder batch: currencies of mid-tier countries ───────────────────────
    {"id": "cur_th", "category": "misc", "question": "What is the official currency of Thailand?", "answers": ["baht"]},
    {"id": "cur_vn", "category": "misc", "question": "What is the official currency of Vietnam?", "answers": ["dong", "đồng"]},
    {"id": "cur_kr", "category": "misc", "question": "What is the official currency of South Korea?", "answers": ["won"]},
    {"id": "cur_cz", "category": "misc", "question": "What is the official currency of the Czech Republic?", "answers": ["koruna"]},
    {"id": "cur_dk", "category": "misc", "question": "What is the official currency of Denmark?", "answers": ["krone"]},
    {"id": "cur_hu", "category": "misc", "question": "What is the official currency of Hungary?", "answers": ["forint"]},
    {"id": "cur_za", "category": "misc", "question": "What is the official currency of South Africa?", "answers": ["rand"]},

    # ── harder batch: geography superlatives (regional) ──────────────────────
    {"id": "geo_island", "category": "geography", "question": "What is the largest island in the world?", "answers": ["Greenland"]},
    {"id": "geo_medisland", "category": "geography", "question": "What is the largest island in the Mediterranean Sea?", "answers": ["Sicily"]},
    {"id": "geo_africamt", "category": "geography", "question": "What is the highest mountain in Africa?", "answers": ["Kilimanjaro"]},
    {"id": "geo_namt", "category": "geography", "question": "What is the highest mountain in North America?", "answers": ["Denali", "McKinley"]},
    {"id": "geo_samt", "category": "geography", "question": "What is the highest mountain in South America?", "answers": ["Aconcagua"]},
    {"id": "geo_deeplake", "category": "geography", "question": "What is the deepest lake in the world?", "answers": ["Baikal", "Lake Baikal"]},
    {"id": "geo_waterfall", "category": "geography", "question": "What is the highest waterfall in the world?", "answers": ["Angel Falls", "Angel"]},
    {"id": "geo_africabig", "category": "geography", "question": "What is the largest country in Africa by land area?", "answers": ["Algeria"]},
    {"id": "geo_asiarriver", "category": "geography", "question": "What is the longest river in Asia?", "answers": ["Yangtze"]},
    {"id": "geo_uscounty", "category": "geography", "question": "Which U.S. state has the most counties?", "answers": ["Texas"]},

    # ── harder batch: science (elements, biology, physics) ───────────────────
    {"id": "el_hg", "category": "science", "question": "What is the chemical symbol for mercury?", "answers": ["Hg"]},
    {"id": "el_ag", "category": "science", "question": "What is the chemical symbol for silver?", "answers": ["Ag"]},
    {"id": "el_sn", "category": "science", "question": "What is the chemical symbol for tin?", "answers": ["Sn"]},
    {"id": "el_sb", "category": "science", "question": "What is the chemical symbol for antimony?", "answers": ["Sb"]},
    {"id": "el_fe_num", "category": "science", "question": "What is the atomic number of iron?", "answers": ["26"]},
    {"id": "el_u_num", "category": "science", "question": "What is the atomic number of uranium?", "answers": ["92"]},
    {"id": "bio_mito", "category": "science", "question": "Which organelle is known as the powerhouse of the cell?", "answers": ["mitochondria", "mitochondrion"]},
    {"id": "bio_gas", "category": "science", "question": "What is the most abundant gas in Earth's atmosphere?", "answers": ["nitrogen"]},
    {"id": "bio_chrom", "category": "science", "question": "How many pairs of chromosomes does a typical human have?", "answers": ["23", "twenty-three"]},
    {"id": "phy_mendeleev", "category": "science", "question": "Who is known as the father of the periodic table?", "answers": ["Mendeleev", "Dmitri Mendeleev"]},
    {"id": "phy_relativity", "category": "science", "question": "Who developed the theory of general relativity?", "answers": ["Einstein", "Albert Einstein"]},

    # ── harder batch: history / dates / discoverers ──────────────────────────
    {"id": "his_civilwar", "category": "history", "question": "In what year did the American Civil War begin?", "answers": ["1861"]},
    {"id": "his_ussr", "category": "history", "question": "In what year did the Soviet Union dissolve?", "answers": ["1991"]},
    {"id": "his_ww1", "category": "history", "question": "In what year did the First World War begin?", "answers": ["1914"]},
    {"id": "his_chernobyl", "category": "history", "question": "In what year did the Chernobyl nuclear disaster occur?", "answers": ["1986"]},
    {"id": "his_penicillin", "category": "history", "question": "Who discovered penicillin?", "answers": ["Fleming", "Alexander Fleming"]},
    {"id": "his_gagarin", "category": "history", "question": "Who was the first human to travel into space?", "answers": ["Gagarin", "Yuri Gagarin"]},
    {"id": "his_sistine", "category": "history", "question": "Who painted the ceiling of the Sistine Chapel?", "answers": ["Michelangelo"]},
    {"id": "his_mayflower", "category": "history", "question": "Which ship carried the Pilgrims to America in 1620?", "answers": ["Mayflower"]},
    {"id": "his_genghis", "category": "history", "question": "Which empire was founded and ruled by Genghis Khan?", "answers": ["Mongol", "Mongol Empire"]},
]

# Quick integrity check: ids must be unique.
_ids = [p["id"] for p in PROMPTS]
assert len(_ids) == len(set(_ids)), "duplicate prompt id detected"


if __name__ == "__main__":
    from collections import Counter
    print(f"{len(PROMPTS)} prompts across categories:")
    for cat, n in sorted(Counter(p["category"] for p in PROMPTS).items()):
        print(f"  {cat:12s} {n}")
