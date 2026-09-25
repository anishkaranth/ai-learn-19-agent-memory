"""Seeded synthetic conversations: a user mentions personal facts at random turns among small talk
(including distractors that reuse fact keywords), and at the end is asked about each fact."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

# slot -> (statement template, question, value pool)
SLOTS = {
    "dog": ("By the way, my dog is called {v}.", "What is my dog called?", ["Biscuit", "Pepper", "Mango", "Waffles", "Luna", "Rocket"]),
    "job": ("I work as a {v} these days.", "What do I work as?", ["nurse", "carpenter", "data analyst", "pilot", "teacher", "chef"]),
    "city": ("I live in {v} now.", "Which city do I live in?", ["Lisbon", "Osaka", "Denver", "Nairobi", "Tallinn", "Porto"]),
    "sister": ("My sister {v} is visiting next month.", "What is my sister's name?", ["Priya", "Hannah", "Mei", "Sofia", "Amara", "Ingrid"]),
    "allergy": ("I'm allergic to {v}, so be careful with recipes.", "What am I allergic to?", ["peanuts", "shellfish", "sesame", "penicillin", "kiwi", "gluten"]),
    "car": ("I drive a {v}.", "What car do I drive?", ["blue Corolla", "red Civic", "white Golf", "grey Model 3", "green Jazz", "black Outback"]),
    "birthday": ("My birthday is on {v}.", "When is my birthday?", ["March 3", "July 19", "October 27", "January 8", "May 14", "December 2"]),
    "food": ("My favourite food is {v}.", "What is my favourite food?", ["ramen", "paella", "dosa", "lasagne", "pho", "tacos"]),
    "hobby": ("On weekends I go {v}.", "What do I do on weekends?", ["bouldering", "kayaking", "birdwatching", "trail running", "sailing", "fishing"]),
    "team": ("I support {v} in football.", "Which football team do I support?", ["Arsenal", "Benfica", "Ajax", "Celtic", "Napoli", "Boca Juniors"]),
    "language": ("I'm learning {v} in the evenings.", "Which language am I learning?", ["Portuguese", "Japanese", "Swahili", "Finnish", "Korean", "Greek"]),
    "coffee": ("I take my coffee {v}.", "How do I take my coffee?", ["black", "with oat milk", "with two sugars", "as a flat white", "iced", "with cardamom"]),
}

SMALL_TALK = [
    "The weather has been strange this week.", "I watched a documentary about octopuses last night.",
    "Work was exhausting today.", "My neighbour's dog barked all night.", "Can you recommend a good podcast?",
    "I finally cleaned the garage.", "The traffic in the city was awful this morning.", "I'm thinking about repainting the kitchen.",
    "Do you know any good stretching routines?", "I burned the toast again.", "My phone battery keeps dying.",
    "I found an old photo album yesterday.", "The coffee machine at the office broke.", "I need to renew my passport soon.",
    "What's a quick dinner idea for tonight?", "I started reading a mystery novel.", "My friend's birthday party was fun.",
    "I might take a weekend trip somewhere.", "The football match last night was boring.", "I'm trying to drink more water.",
    "Our team at work got a new manager.", "I keep forgetting to water the plants.", "Is it normal to feel tired after lunch?",
    "The car needs a service soon.", "I tried a new recipe and it was fine.", "I want to learn to play the guitar someday.",
]
AGENT_REPLIES = ["Got it, thanks for telling me.", "That sounds nice.", "I understand.", "Interesting, tell me more.",
                 "Noted.", "Sounds like quite a day.", "Happy to help with that.", "Okay."]


@dataclass
class Turn:
    idx: int
    user: str
    agent: str
    slot: str | None = None


@dataclass
class Probe:
    slot: str
    question: str
    answer: str
    fact_turn: int


def make_episode(rng: random.Random, n_turns: int = 80):
    slots = list(SLOTS)
    fact_turns = dict(zip(rng.sample(range(n_turns), len(slots)), slots))
    turns, probes = [], []
    for t in range(n_turns):
        slot = fact_turns.get(t)
        if slot:
            tmpl, q, pool = SLOTS[slot]
            v = rng.choice(pool)
            user = tmpl.format(v=v)
            probes.append(Probe(slot, q, v, t))
        else:
            user = rng.choice(SMALL_TALK)
        turns.append(Turn(t, user, rng.choice(AGENT_REPLIES), slot))
    return turns, probes


def make_episodes(n: int = 40, n_turns: int = 80, seed: int = 42):
    rng = random.Random(seed)
    return [make_episode(rng, n_turns) for _ in range(n)]
