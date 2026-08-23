"""A third prompt bank: single-fact recall, chosen to be hard for a 1B model.

Why this exists
---------------
`common.prompts` (the `factual` bank) is too easy to measure anything with: on
llama, the 1B scores 88.6% against the 3B's 97.1%, leaving 8.6 points of
headroom, of which 3 prompts on a 35-prompt split are actually divergent. A
large->small stitch is trying to buy the small model's *wrong* answers back, so
a bank where the small model is already right has nothing to sell.

`common.prompts_list` solves that a different way — conjunctive multi-item
answers — and gets 17.8 points on llama. But its answers run 10-25 tokens and
are scored all-or-nothing across items, which mixes "did the large model's
knowledge transfer" with "did the small model's late blocks keep a long list
straight". For an accuracy experiment we want the first question isolated.

So this bank keeps answers *short* (a name, a symbol, a number) and buys
difficulty from obscurity instead of from length:

* capitals of countries people cannot name offhand, not of France
* every US state capital, including the ones that are not the largest city
* element symbols that are not the mnemonic-famous ones
* currencies of mid-tier economies
* years of events that are dated but not iconic

That is the regime where a 1B model confabulates a plausible-looking answer
(`"Igor's Falls"`, `"121 keys"`, `Auckland` for Wellington) and a 3B model
still recalls the fact — exactly the small-wrong/large-right cases a
large->small stitch needs in order to have something to recover.

Scoring is ANY-alias (`common.scoring.is_correct`), so spelling and transliteration
variants score the same. Facts are textbook-stable; nothing here depends on a
current office-holder, a population figure, or a record that changes.

Auditing
--------
The uniform categories are stored as one-line `"question-key -> answer"` tables
and expanded into prompt dicts below, so a reviewer checks a fact by reading one
aligned line rather than a five-line dict. Aliases are `|`-separated.

`python -m stitching_large_to_small.run headroom` reports every prompt that BOTH
models get wrong; those are the prime suspects for a bad gold answer here, and
should be audited before any accuracy claim rests on them.
"""

# ── national capitals ─────────────────────────────────────────────────────────
# Deliberately skewed away from the famous ones: a 1B model answers "France" and
# "Japan" perfectly well, so those prompts only dilute the signal.
CAPITALS = """
Bhutan -> Thimphu
Mongolia -> Ulaanbaatar|Ulan Bator
Kazakhstan -> Astana|Nur-Sultan|Nur Sultan
Myanmar -> Naypyidaw|Naypyitaw|Nay Pyi Taw
Suriname -> Paramaribo
Guyana -> Georgetown
Paraguay -> Asuncion|Asunción
Uruguay -> Montevideo
Bolivia -> Sucre|La Paz
Ecuador -> Quito
Nicaragua -> Managua
Honduras -> Tegucigalpa
El Salvador -> San Salvador
Costa Rica -> San Jose|San José
Panama -> Panama City
Belize -> Belmopan
Jamaica -> Kingston
Haiti -> Port-au-Prince
Dominican Republic -> Santo Domingo
Trinidad and Tobago -> Port of Spain|Port-of-Spain
Senegal -> Dakar
Mali -> Bamako
Burkina Faso -> Ouagadougou
Niger -> Niamey
Chad -> N'Djamena|Ndjamena
Cameroon -> Yaounde|Yaoundé
Gabon -> Libreville
Angola -> Luanda
Zambia -> Lusaka
Zimbabwe -> Harare
Botswana -> Gaborone
Namibia -> Windhoek
Mozambique -> Maputo
Madagascar -> Antananarivo
Tanzania -> Dodoma
Uganda -> Kampala
Rwanda -> Kigali
Ethiopia -> Addis Ababa
Sudan -> Khartoum
Ghana -> Accra
Ivory Coast -> Yamoussoukro
Tunisia -> Tunis
Libya -> Tripoli
Morocco -> Rabat
Albania -> Tirana|Tirane
Slovenia -> Ljubljana
Slovakia -> Bratislava
Croatia -> Zagreb
Serbia -> Belgrade
Bulgaria -> Sofia
Romania -> Bucharest
Moldova -> Chisinau|Chișinău|Kishinev
Belarus -> Minsk
Latvia -> Riga
Lithuania -> Vilnius
Estonia -> Tallinn
Georgia -> Tbilisi
Armenia -> Yerevan
Azerbaijan -> Baku
Uzbekistan -> Tashkent
Turkmenistan -> Ashgabat
Kyrgyzstan -> Bishkek
Tajikistan -> Dushanbe
Nepal -> Kathmandu
Sri Lanka -> Sri Jayawardenepura Kotte|Colombo
Bangladesh -> Dhaka
Cambodia -> Phnom Penh
Laos -> Vientiane
Malaysia -> Kuala Lumpur
Brunei -> Bandar Seri Begawan
Papua New Guinea -> Port Moresby
Fiji -> Suva
Qatar -> Doha
Oman -> Muscat
Jordan -> Amman
Lebanon -> Beirut
Yemen -> Sanaa|Sana'a
Iceland -> Reykjavik|Reykjavík
"""

# ── US state capitals ─────────────────────────────────────────────────────────
# The classic confabulation trap: for most states the capital is not the largest
# city, and a small model reaches for the largest city.
US_CAPITALS = """
Alabama -> Montgomery
Alaska -> Juneau
Arizona -> Phoenix
Arkansas -> Little Rock
California -> Sacramento
Colorado -> Denver
Connecticut -> Hartford
Delaware -> Dover
Florida -> Tallahassee
Georgia -> Atlanta
Hawaii -> Honolulu
Idaho -> Boise
Illinois -> Springfield
Indiana -> Indianapolis
Iowa -> Des Moines
Kansas -> Topeka
Kentucky -> Frankfort
Louisiana -> Baton Rouge
Maine -> Augusta
Maryland -> Annapolis
Massachusetts -> Boston
Michigan -> Lansing
Minnesota -> Saint Paul|St. Paul|St Paul
Mississippi -> Jackson
Missouri -> Jefferson City
Montana -> Helena
Nebraska -> Lincoln
Nevada -> Carson City
New Hampshire -> Concord
New Jersey -> Trenton
New Mexico -> Santa Fe
New York -> Albany
North Carolina -> Raleigh
North Dakota -> Bismarck
Ohio -> Columbus
Oklahoma -> Oklahoma City
Oregon -> Salem
Pennsylvania -> Harrisburg
Rhode Island -> Providence
South Carolina -> Columbia
South Dakota -> Pierre
Tennessee -> Nashville
Texas -> Austin
Utah -> Salt Lake City
Vermont -> Montpelier
Virginia -> Richmond
Washington -> Olympia
West Virginia -> Charleston
Wisconsin -> Madison
Wyoming -> Cheyenne
"""

# ── element symbols ───────────────────────────────────────────────────────────
# Skewed to the ones whose symbol comes from Latin or is otherwise not the first
# letters of the English name.
ELEMENTS = """
tungsten -> W
antimony -> Sb
tin -> Sn
lead -> Pb
mercury -> Hg
silver -> Ag
gold -> Au
copper -> Cu
iron -> Fe
sodium -> Na
potassium -> K
manganese -> Mn
magnesium -> Mg
molybdenum -> Mo
zirconium -> Zr
tellurium -> Te
selenium -> Se
arsenic -> As
bismuth -> Bi
cadmium -> Cd
cobalt -> Co
chromium -> Cr
nickel -> Ni
niobium -> Nb
palladium -> Pd
platinum -> Pt
rhodium -> Rh
ruthenium -> Ru
strontium -> Sr
barium -> Ba
beryllium -> Be
boron -> B
bromine -> Br
fluorine -> F
iodine -> I
krypton -> Kr
xenon -> Xe
argon -> Ar
neon -> Ne
titanium -> Ti
vanadium -> V
zinc -> Zn
uranium -> U
thorium -> Th
"""

# ── atomic numbers ────────────────────────────────────────────────────────────
ATOMIC_NUMBERS = """
hydrogen -> 1
helium -> 2
lithium -> 3
beryllium -> 4
boron -> 5
carbon -> 6
nitrogen -> 7
oxygen -> 8
sodium -> 11
magnesium -> 12
aluminium -> 13
silicon -> 14
phosphorus -> 15
sulfur -> 16
chlorine -> 17
potassium -> 19
calcium -> 20
titanium -> 22
chromium -> 24
iron -> 26
cobalt -> 27
nickel -> 28
copper -> 29
zinc -> 30
silver -> 47
tin -> 50
iodine -> 53
gold -> 79
mercury -> 80
lead -> 82
uranium -> 92
fluorine -> 9
neon -> 10
argon -> 18
scandium -> 21
vanadium -> 23
manganese -> 25
gallium -> 31
germanium -> 32
arsenic -> 33
selenium -> 34
bromine -> 35
krypton -> 36
rubidium -> 37
strontium -> 38
yttrium -> 39
zirconium -> 40
niobium -> 41
molybdenum -> 42
ruthenium -> 44
rhodium -> 45
palladium -> 46
cadmium -> 48
indium -> 49
antimony -> 51
tellurium -> 52
xenon -> 54
caesium -> 55
barium -> 56
lanthanum -> 57
tungsten -> 74
osmium -> 76
iridium -> 77
platinum -> 78
thallium -> 81
bismuth -> 83
radon -> 86
radium -> 88
thorium -> 90
plutonium -> 94
"""

# ── currencies ────────────────────────────────────────────────────────────────
CURRENCIES = """
Poland -> zloty|złoty
Hungary -> forint
Czech Republic -> koruna
Sweden -> krona|kronor
Denmark -> krone
Norway -> krone
Iceland -> krona|króna
Switzerland -> franc
Turkey -> lira
Israel -> shekel
Saudi Arabia -> riyal
Qatar -> riyal
Oman -> rial
Kuwait -> dinar
Iraq -> dinar
Jordan -> dinar
Serbia -> dinar
Algeria -> dinar
Tunisia -> dinar
Morocco -> dirham
United Arab Emirates -> dirham
India -> rupee|rupees
Pakistan -> rupee|rupees
Nepal -> rupee|rupees
Indonesia -> rupiah
Malaysia -> ringgit
Thailand -> baht
Vietnam -> dong|đồng
South Korea -> won
North Korea -> won
Philippines -> peso
Mexico -> peso
Chile -> peso
Colombia -> peso
Argentina -> peso
Brazil -> real
Peru -> sol
Venezuela -> bolivar|bolívar
Costa Rica -> colon|colón
Guatemala -> quetzal
Panama -> balboa|dollar|US dollar
Nigeria -> naira
Ghana -> cedi
Kenya -> shilling
Tanzania -> shilling
Ethiopia -> birr
South Africa -> rand
Zambia -> kwacha
Botswana -> pula
Angola -> kwanza
Russia -> ruble|rouble
Ukraine -> hryvnia
Kazakhstan -> tenge
Mongolia -> tugrik|tögrög|togrog
Bangladesh -> taka
Sri Lanka -> rupee|rupees
Myanmar -> kyat
Cambodia -> riel
Laos -> kip
"""

# ── highest peak by country ───────────────────────────────────────────────────
HIGHEST_PEAKS = """
Africa -> Kilimanjaro
North America -> Denali|McKinley
South America -> Aconcagua
Europe -> Elbrus|Mont Blanc
Antarctica -> Vinson|Mount Vinson
Australia -> Kosciuszko|Mount Kosciuszko
Japan -> Fuji|Mount Fuji
Tanzania -> Kilimanjaro
Argentina -> Aconcagua
Nepal -> Everest|Mount Everest
"""

# ── longest river on each continent ───────────────────────────────────────────
LONGEST_RIVERS = """
Africa -> Nile
South America -> Amazon
Asia -> Yangtze
Europe -> Volga
North America -> Missouri|Mississippi
Australia -> Murray
"""


def _rows(table: str):
    for line in table.strip().splitlines():
        key, _, answer = line.partition("->")
        yield key.strip(), [a.strip() for a in answer.split("|") if a.strip()]


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in text]
    return "".join(keep).strip("_").replace("__", "_")[:24]


PROMPTS: list[dict] = []


def _add(prefix: str, category: str, table: str, question: str,
         capitalize: bool = False) -> None:
    """Expand a fact table into prompt dicts.

    `capitalize` upper-cases the key's first letter, for tables whose keys are
    clauses that open the question rather than nouns dropped into the middle
    of it.
    """
    for key, answers in _rows(table):
        shown = key[0].upper() + key[1:] if capitalize else key
        PROMPTS.append({
            "id": f"{prefix}_{_slug(key)}",
            "category": category,
            "question": question.format(key=shown),
            "answers": answers,
        })


_add("hcap", "capital", CAPITALS, "What is the capital city of {key}?")
_add("hus", "us_capital", US_CAPITALS,
     "What is the capital of the U.S. state of {key}?")
_add("hsym", "element", ELEMENTS, "What is the chemical symbol for {key}?")
_add("hnum", "atomic_number", ATOMIC_NUMBERS, "What is the atomic number of {key}?")
_add("hcur", "currency", CURRENCIES, "What is the official currency of {key}?")
_add("hpeak", "geography", HIGHEST_PEAKS, "What is the highest mountain in {key}?")
_add("hriv", "geography", LONGEST_RIVERS, "What is the longest river in {key}?")


# ── dated events ──────────────────────────────────────────────────────────────
# Measured at 67% divergence on llama/dev: the 1B reliably produces a
# plausible-looking wrong year (1980 for the Berlin Wall, 1959 for penicillin)
# while the 3B recalls it. The single most productive category in this bank.
EVENT_YEARS = """
the Magna Carta was sealed -> 1215
Constantinople fell to the Ottomans -> 1453
Martin Luther published his Ninety-five Theses -> 1517
the Mayflower carried the Pilgrims to America -> 1620
the Great Fire of London happened -> 1666
the American Declaration of Independence was signed -> 1776
the Louisiana Purchase was completed -> 1803
slavery was abolished throughout the British Empire -> 1833
the American Civil War began -> 1861
Abraham Lincoln was assassinated -> 1865
Alexander Graham Bell patented the telephone -> 1876
the Wright brothers made their first powered flight -> 1903
the Russian Revolution took place -> 1917
the Wall Street Crash began the Great Depression -> 1929
the Second World War began in Europe -> 1939
Pearl Harbor was attacked -> 1941
the D-Day landings took place in Normandy -> 1944
Mount Everest was first summited -> 1953
Sputnik 1 was launched -> 1957
the Cuban Missile Crisis took place -> 1962
John F. Kennedy was assassinated -> 1963
the Channel Tunnel opened -> 1994
the first iPhone was released -> 2007
the Hubble Space Telescope was launched -> 1990
the Chernobyl disaster occurred -> 1986
Nelson Mandela was released from prison -> 1990
the Titanic sank -> 1912
the Berlin Olympics were held -> 1936
the League of Nations was founded -> 1920
the first modern Olympic Games were held in Athens -> 1896
"""

_add("hev", "history", EVENT_YEARS, "{key}. In what year did this happen?",
     capitalize=True)


# ── one-off facts that do not fit a table ─────────────────────────────────────
# Written out because each needs its own phrasing; same shape as the generated
# entries. Chosen on the same principle: dated, stable, and not iconic.
PROMPTS += [
    # history — years that are precise but not famous
    {"id": "hyr_versailles", "category": "history", "question": "In what year was the Treaty of Versailles signed?", "answers": ["1919"]},
    {"id": "hyr_hastings", "category": "history", "question": "In what year was the Battle of Hastings?", "answers": ["1066"]},
    {"id": "hyr_printing", "category": "history", "question": "In what decade did Gutenberg print his famous Bible? Answer with the year, approximately.", "answers": ["1455", "1450s", "1454", "1456"]},
    {"id": "hyr_armada", "category": "history", "question": "In what year was the Spanish Armada defeated?", "answers": ["1588"]},
    {"id": "hyr_waterloo", "category": "history", "question": "In what year was the Battle of Waterloo?", "answers": ["1815"]},
    {"id": "hyr_boston_tea", "category": "history", "question": "In what year was the Boston Tea Party?", "answers": ["1773"]},
    {"id": "hyr_apollo13", "category": "history", "question": "In what year did the Apollo 13 mission take place?", "answers": ["1970"]},
    {"id": "hyr_suez", "category": "history", "question": "In what year did the Suez Canal open?", "answers": ["1869"]},
    {"id": "hyr_panama", "category": "history", "question": "In what year did the Panama Canal open?", "answers": ["1914"]},
    {"id": "hyr_titanic_yr", "category": "history", "question": "In what year was the Eiffel Tower completed?", "answers": ["1889"]},
    {"id": "hyr_penicillin", "category": "history", "question": "In what year did Alexander Fleming discover penicillin?", "answers": ["1928"]},
    {"id": "hyr_dna", "category": "history", "question": "In what year was the structure of DNA published by Watson and Crick?", "answers": ["1953"]},
    {"id": "hyr_wall_built", "category": "history", "question": "In what year was the Berlin Wall built?", "answers": ["1961"]},
    {"id": "hyr_un", "category": "history", "question": "In what year was the United Nations founded?", "answers": ["1945"]},
    {"id": "hyr_nato", "category": "history", "question": "In what year was NATO founded?", "answers": ["1949"]},

    # science — stable constants and canonical facts
    {"id": "hsci_absolute", "category": "science", "question": "What is absolute zero in degrees Celsius, rounded to the nearest whole number?", "answers": ["-273", "273"]},
    {"id": "hsci_freeze_f", "category": "science", "question": "At what temperature in degrees Fahrenheit does water freeze?", "answers": ["32"]},
    {"id": "hsci_boil_f", "category": "science", "question": "At what temperature in degrees Fahrenheit does water boil at sea level?", "answers": ["212"]},
    {"id": "hsci_ph_neutral", "category": "science", "question": "What is the pH of a neutral solution at 25 degrees Celsius?", "answers": ["7"]},
    {"id": "hsci_avogadro", "category": "science", "question": "What is Avogadro's number, to three significant figures?", "answers": ["6.02", "6.022"]},
    {"id": "hsci_planets_rings", "category": "science", "question": "Which planet is known as the Red Planet?", "answers": ["Mars"]},
    {"id": "hsci_largest_planet", "category": "science", "question": "Which is the largest planet in the Solar System?", "answers": ["Jupiter"]},
    {"id": "hsci_hottest_planet", "category": "science", "question": "Which planet in the Solar System has the highest average surface temperature?", "answers": ["Venus"]},
    {"id": "hsci_moon_mars", "category": "science", "question": "How many moons does Mars have?", "answers": ["2", "two"]},
    {"id": "hsci_bone_longest", "category": "science", "question": "What is the longest bone in the human body?", "answers": ["femur", "thigh bone"]},
    {"id": "hsci_largest_organ", "category": "science", "question": "What is the largest organ of the human body?", "answers": ["skin"]},
    {"id": "hsci_blood_cells", "category": "science", "question": "Which blood cells are primarily responsible for fighting infection?", "answers": ["white blood cells", "leukocytes", "leucocytes", "neutrophils", "neutrophil", "lymphocytes"]},
    {"id": "hsci_heart_chambers", "category": "science", "question": "How many chambers does the human heart have?", "answers": ["4", "four"]},
    {"id": "hsci_ribs", "category": "science", "question": "How many pairs of ribs does a typical human have?", "answers": ["12", "twelve"]},
    {"id": "hsci_teeth", "category": "science", "question": "How many teeth does a typical adult human have?", "answers": ["32", "thirty-two"]},
    {"id": "hsci_speed_sound", "category": "science", "question": "What is the approximate speed of sound in air at sea level, in metres per second?", "answers": ["343", "340", "330", "331"]},
    {"id": "hsci_light_year", "category": "science", "question": "What unit of distance is defined as the distance light travels in one year?", "answers": ["light year", "light-year"]},
    {"id": "hsci_salt", "category": "science", "question": "What is the chemical formula of table salt?", "answers": ["NaCl"]},
    {"id": "hsci_methane", "category": "science", "question": "What is the chemical formula of methane?", "answers": ["CH4"]},
    {"id": "hsci_ammonia", "category": "science", "question": "What is the chemical formula of ammonia?", "answers": ["NH3"]},
    {"id": "hsci_sulfuric", "category": "science", "question": "What is the chemical formula of sulfuric acid?", "answers": ["H2SO4"]},
    {"id": "hsci_glucose", "category": "science", "question": "What is the chemical formula of glucose?", "answers": ["C6H12O6"]},
    {"id": "hsci_hardness", "category": "science", "question": "What scale is used to measure the hardness of minerals?", "answers": ["Mohs", "Mohs scale"]},
    {"id": "hsci_richter", "category": "science", "question": "Which scale was historically used to measure earthquake magnitude?", "answers": ["Richter", "Richter scale"]},
    {"id": "hsci_photosynth", "category": "science", "question": "What pigment makes plants green and captures light for photosynthesis?", "answers": ["chlorophyll"]},

    # literature and art — attribution of works that are known but not top-ten
    {"id": "hlit_ulysses", "category": "literature", "question": "Who wrote the novel 'Ulysses'?", "answers": ["Joyce", "James Joyce"]},
    {"id": "hlit_crime", "category": "literature", "question": "Who wrote the novel 'Crime and Punishment'?", "answers": ["Dostoevsky", "Fyodor Dostoevsky", "Dostoyevsky"]},
    {"id": "hlit_metamorph", "category": "literature", "question": "Who wrote 'The Metamorphosis'?", "answers": ["Kafka", "Franz Kafka"]},
    {"id": "hlit_gatsby", "category": "literature", "question": "Who wrote 'The Great Gatsby'?", "answers": ["Fitzgerald", "F. Scott Fitzgerald"]},
    {"id": "hlit_mockingbird", "category": "literature", "question": "Who wrote 'To Kill a Mockingbird'?", "answers": ["Harper Lee", "Lee"]},
    {"id": "hlit_brave", "category": "literature", "question": "Who wrote 'Brave New World'?", "answers": ["Huxley", "Aldous Huxley"]},
    {"id": "hlit_moby", "category": "literature", "question": "Who wrote 'Moby-Dick'?", "answers": ["Melville", "Herman Melville"]},
    {"id": "hlit_hundred", "category": "literature", "question": "Who wrote 'One Hundred Years of Solitude'?", "answers": ["Garcia Marquez", "Gabriel Garcia Marquez", "Márquez", "García Márquez"]},
    {"id": "hlit_divine", "category": "literature", "question": "Who wrote 'The Divine Comedy'?", "answers": ["Dante", "Dante Alighieri"]},
    {"id": "hlit_faust", "category": "literature", "question": "Who wrote 'Faust'?", "answers": ["Goethe", "Johann Wolfgang von Goethe"]},
    {"id": "hart_scream", "category": "literature", "question": "Who painted 'The Scream'?", "answers": ["Munch", "Edvard Munch"]},
    {"id": "hart_persistence", "category": "literature", "question": "Who painted 'The Persistence of Memory'?", "answers": ["Dali", "Salvador Dali", "Dalí"]},
    {"id": "hart_nightwatch", "category": "literature", "question": "Who painted 'The Night Watch'?", "answers": ["Rembrandt"]},
    {"id": "hart_birth_venus", "category": "literature", "question": "Who painted 'The Birth of Venus'?", "answers": ["Botticelli", "Sandro Botticelli"]},
    {"id": "hart_pearl", "category": "literature", "question": "Who painted 'Girl with a Pearl Earring'?", "answers": ["Vermeer", "Johannes Vermeer"]},
    {"id": "hart_thinker", "category": "literature", "question": "Which sculptor created 'The Thinker'?", "answers": ["Rodin", "Auguste Rodin"]},
    {"id": "hart_david", "category": "literature", "question": "Which sculptor created the statue of David in Florence?", "answers": ["Michelangelo"]},

    # misc numeric and definitional facts
    {"id": "hmis_chess_pieces", "category": "misc", "question": "How many pieces does each player start with in chess?", "answers": ["16", "sixteen"]},
    {"id": "hmis_sonnet", "category": "misc", "question": "How many lines are in a sonnet?", "answers": ["14", "fourteen"]},
    {"id": "hmis_haiku", "category": "misc", "question": "How many syllables in total does a traditional haiku have across all three lines?", "answers": ["17", "seventeen"]},
    {"id": "hmis_olympic_rings", "category": "misc", "question": "How many rings are on the Olympic flag?", "answers": ["5", "five"]},
    {"id": "hmis_zodiac", "category": "misc", "question": "How many signs are in the Western zodiac?", "answers": ["12", "twelve"]},
    {"id": "hmis_deck", "category": "misc", "question": "How many cards are in a standard deck without jokers?", "answers": ["52", "fifty-two"]},
    {"id": "hmis_soccer_players", "category": "misc", "question": "How many players are on the field per team in a football (soccer) match?", "answers": ["11", "eleven"]},
    {"id": "hmis_basketball", "category": "misc", "question": "How many players are on the court per team in basketball?", "answers": ["5", "five"]},
    {"id": "hmis_holes_golf", "category": "misc", "question": "How many holes are on a standard golf course?", "answers": ["18", "eighteen"]},
    {"id": "hmis_marathon_miles", "category": "misc", "question": "How many miles long is a marathon, to one decimal place?", "answers": ["26.2"]},
    {"id": "hmis_greek_letters", "category": "misc", "question": "How many letters are in the Greek alphabet?", "answers": ["24", "twenty-four"]},
    {"id": "hmis_amendments_bill", "category": "misc", "question": "How many amendments make up the United States Bill of Rights?", "answers": ["10", "ten"]},
    {"id": "hmis_senators", "category": "misc", "question": "How many senators does each U.S. state have?", "answers": ["2", "two"]},
    {"id": "hmis_eu_founding", "category": "misc", "question": "How many countries signed the 1957 Treaty of Rome founding the EEC?", "answers": ["6", "six"]},
    {"id": "hmis_lang_brazil", "category": "misc", "question": "What is the official language of Angola?", "answers": ["Portuguese"]},
    {"id": "hmis_lang_suriname", "category": "misc", "question": "What is the official language of Suriname?", "answers": ["Dutch"]},
    {"id": "hmis_lang_philippines", "category": "misc", "question": "Besides English, what is the other official language of the Philippines?", "answers": ["Filipino", "Tagalog"]},
    {"id": "hmis_lang_egypt", "category": "misc", "question": "What is the official language of Egypt?", "answers": ["Arabic"]},
]

# Quick integrity check: ids must be unique and every item must be answerable.
_ids = [p["id"] for p in PROMPTS]
assert len(_ids) == len(set(_ids)), \
    f"duplicate prompt id: {sorted({i for i in _ids if _ids.count(i) > 1})}"
assert all(p["answers"] for p in PROMPTS), "an item has no accepted answers"

HARD_FACTUAL_PROMPTS = PROMPTS


if __name__ == "__main__":
    from collections import Counter
    print(f"{len(PROMPTS)} prompts across categories:")
    for cat, n in sorted(Counter(p["category"] for p in PROMPTS).items()):
        print(f"  {cat:16s} {n}")
