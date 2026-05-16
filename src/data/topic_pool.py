"""Topic pool for shared-prefix conversation generation.

200+ topics across diverse domains, partitioned into non-overlapping
train/val/test splits so no topic leaks across splits.
"""

import random

TOPICS = {
    "consumer_lifestyle": [
        "planning a weekend hiking trip",
        "choosing a new laptop for college",
        "learning to cook Italian food",
        "training for a half marathon",
        "starting a small vegetable garden",
        "picking a book for a reading club",
        "planning a budget-friendly vacation",
        "learning basic photography techniques",
        "setting up a home office workspace",
        "planning a birthday party for a friend",
        "understanding how to invest in index funds",
        "learning to play the guitar",
        "organizing a cluttered garage",
        "adopting a rescue dog",
        "switching to a plant-based diet",
        "building a personal website",
        "preparing for a job interview",
        "improving public speaking skills",
        "choosing paint colors for a room",
        "understanding nutrition labels on food",
        "planning a road trip across the country",
        "setting up a fish tank",
        "learning basic car maintenance",
        "starting a podcast",
        "understanding different types of tea",
        "helping a kid with math homework",
        "choosing between streaming services",
        "learning to sew and mend clothes",
        "planning meals for the week",
        "choosing a new phone plan",
        "understanding how solar panels work",
        "getting into board games",
        "learning basic first aid",
        "choosing a mattress",
        "understanding pet insurance",
        "learning about wine pairing",
        "choosing a daycare for a toddler",
        "fixing a leaky faucet",
        "understanding credit scores",
        "planning a camping trip with kids",
        "choosing house plants for low light",
        "understanding different yoga styles",
        "shopping for running shoes",
        "learning to make sourdough bread",
        "understanding health insurance options",
        "choosing a bicycle for commuting",
        "learning about composting at home",
        "choosing a water filter for the kitchen",
        "getting started with meditation",
        "choosing a gift for a coworker",
    ],
    "professional_technical": [
        "choosing a programming language to learn",
        "setting up a CI/CD pipeline for a small team",
        "understanding Kubernetes pod networking",
        "migrating a database from MySQL to PostgreSQL",
        "choosing between REST and GraphQL for an API",
        "understanding OAuth 2.0 authorization flows",
        "planning a microservices architecture transition",
        "debugging memory leaks in a Node.js application",
        "choosing a cloud provider for a startup",
        "understanding container orchestration options",
        "planning sprint retrospectives effectively",
        "choosing a project management methodology",
        "understanding Agile estimation techniques",
        "setting up monitoring and alerting for production",
        "planning a technical interview process",
        "choosing between SQL and NoSQL databases",
        "understanding load balancing strategies",
        "planning a data migration strategy",
        "setting up code review best practices",
        "understanding infrastructure as code tools",
        "choosing a front-end framework for a new project",
        "planning a disaster recovery strategy",
        "understanding API rate limiting approaches",
        "setting up automated testing pipelines",
        "choosing a message queue for async processing",
        "understanding caching strategies for web apps",
        "planning a mobile app architecture",
        "choosing a logging and observability stack",
        "understanding feature flagging systems",
        "planning a technical documentation system",
    ],
    "academic_research": [
        "understanding systematic literature review methods",
        "choosing a statistical analysis approach for survey data",
        "planning a research proposal for a masters thesis",
        "understanding IRB approval requirements",
        "choosing between qualitative and quantitative methods",
        "understanding p-values and statistical significance",
        "planning a longitudinal study design",
        "choosing a citation management tool",
        "understanding research ethics for human subjects",
        "planning an experiment with control groups",
        "understanding meta-analysis techniques",
        "choosing a data visualization approach for a paper",
        "understanding peer review processes",
        "planning a conference presentation",
        "choosing between parametric and non-parametric tests",
        "understanding sampling methods for social science",
        "planning a mixed-methods research design",
        "choosing open access publishing options",
        "understanding research data management plans",
        "planning a collaborative research project",
        "choosing an appropriate effect size measure",
        "understanding grounded theory methodology",
        "planning academic grant applications",
        "choosing software for qualitative data analysis",
        "understanding cross-sectional vs longitudinal designs",
        "planning a thesis defense presentation",
        "choosing appropriate control variables",
        "understanding research reproducibility practices",
        "planning a field study in education research",
        "choosing between case study and survey approaches",
    ],
    "creative_arts": [
        "learning digital illustration techniques",
        "choosing a camera for street photography",
        "understanding color theory for graphic design",
        "planning a short film production",
        "learning music production with a DAW",
        "choosing materials for oil painting",
        "understanding typography for web design",
        "planning a creative writing workshop",
        "learning 3D modeling for beginners",
        "choosing a pottery wheel for home use",
        "understanding composition rules in photography",
        "planning a community mural project",
        "learning hand lettering techniques",
        "choosing instruments for a home studio",
        "understanding print vs digital design workflows",
        "planning an art exhibition",
        "learning stop-motion animation basics",
        "choosing fabric for a quilting project",
        "understanding sound design for podcasts",
        "planning a photography portfolio",
    ],
    "practical_home": [
        "understanding different types of mortgages",
        "planning a kitchen renovation on a budget",
        "choosing energy-efficient appliances",
        "understanding home electrical wiring basics",
        "planning a bathroom remodel",
        "choosing the right insulation for an attic",
        "understanding plumbing fixture options",
        "planning a backyard landscaping project",
        "choosing a home security camera system",
        "understanding property tax assessment appeals",
        "planning a deck or patio construction",
        "choosing exterior paint for different climates",
        "understanding HVAC system maintenance",
        "planning a garage conversion project",
        "choosing smart home automation devices",
        "understanding radon testing and mitigation",
        "planning a fence installation",
        "choosing window treatments for energy efficiency",
        "understanding roof replacement options",
        "planning a closet organization system",
    ],
    "health_wellness": [
        "understanding different types of physical therapy",
        "choosing a fitness tracker for health monitoring",
        "planning a post-surgery recovery routine",
        "understanding food allergy testing options",
        "choosing between different sleep improvement methods",
        "understanding mental health therapy approaches",
        "planning a nutrition plan for athletic training",
        "choosing ergonomic office furniture",
        "understanding cholesterol management through diet",
        "planning a stress management routine",
        "choosing a dentist for orthodontic work",
        "understanding physical rehabilitation exercises",
        "planning a safe exercise routine after injury",
        "choosing supplements based on blood work",
        "understanding mindfulness-based stress reduction",
        "planning vision care and eye exam schedule",
        "choosing a chiropractor vs physical therapist",
        "understanding prenatal care options",
        "planning a balanced meal prep routine",
        "choosing between different allergy treatment options",
    ],
    "finance_business": [
        "understanding small business tax deductions",
        "choosing a business banking account",
        "planning a retirement savings strategy",
        "understanding estate planning basics",
        "choosing between different types of business insurance",
        "planning a budget for a home purchase down payment",
        "understanding 529 education savings plans",
        "choosing a payroll processing system",
        "planning for self-employment taxes",
        "understanding commercial lease negotiations",
        "choosing accounting software for freelancers",
        "planning a debt payoff strategy",
        "understanding Roth IRA conversion rules",
        "choosing a financial advisor",
        "planning inventory management for a retail store",
        "understanding business entity types for liability",
        "choosing a point of sale system",
        "planning cash flow management for seasonal business",
        "understanding angel investing vs venture capital",
        "choosing between different merchant payment processors",
    ],
}

ALL_TOPICS = []
for category, topics in TOPICS.items():
    ALL_TOPICS.extend([(t, category) for t in topics])


def partition_topics(seed=42):
    """Partition topics into non-overlapping train/val/test splits.

    Split: ~70% train, ~15% val, ~15% test.
    No topic appears in more than one split.

    Returns:
        dict with 'train', 'val', 'test' keys, each a list of topic strings.
    """
    rng = random.Random(seed)
    shuffled = list(ALL_TOPICS)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    return {
        "train": [t for t, _ in shuffled[:train_end]],
        "val": [t for t, _ in shuffled[train_end:val_end]],
        "test": [t for t, _ in shuffled[val_end:]],
    }


def get_topic_stats():
    """Return summary statistics about the topic pool."""
    stats = {"total": len(ALL_TOPICS)}
    for category, topics in TOPICS.items():
        stats[category] = len(topics)
    return stats
