# User Interface Design Goals

**Note:** This is a Discord bot, so the primary UI is Discord's native interface. These goals apply to the bot's interaction patterns and command design.

## Overall UX Vision

The bot should feel like a knowledgeable lore expert engaging in natural conversation, responding in Hungarian by default or matching the user's query language (English). Users should not need to learn complex syntax - simple mentions and conversational questions should work seamlessly. The bot's personality mode is configured via .env (e.g., neutral informative vs. grimdark narrator).

## Key Interaction Paradigms

- **Natural Language First:** Users ask questions naturally, no rigid command syntax required
- **Mention-Based Activation:** `@WH40K-Bot Who is Roboute Guilliman?` triggers a query
- **Slash Commands for Features:** `/trivia new` (admin), `/leaderboard` for structured features
- **Interactive Buttons:** Trivia questions use Discord Embed buttons for answer selection
- **Contextual Follow-Ups:** Bot remembers recent conversation context for follow-up questions
- **Inline Citations:** Responses include source article names for user verification

## Core Interaction Flows

1. **Lore Query Flow (Hungarian):**
   - User: `@WH40K-Bot Mesélj a Horus Eretnekségről`
   - Bot: `[Typing indicator]` → Response with citations in Hungarian
   - User: `Mikor történt?` (follow-up)
   - Bot: Responds using conversation context in Hungarian

1a. **Lore Query Flow (English):**
   - User: `@WH40K-Bot Tell me about the Horus Heresy`
   - Bot: `[Typing indicator]` → Response with citations in English
   - User: `When did it happen?` (follow-up)
   - Bot: Responds using conversation context in English

2. **Trivia Flow:**
   - Admin: `/trivia new medium` (starts new trivia question)
   - Bot: Posts Discord Embed with question text and interactive buttons (A, B, C, D)
   - User: Clicks button B
   - Bot: Updates embed with result (correct/incorrect), updates score, shows leaderboard position

3. **Admin Commands:**
   - Admin (with configured Discord role): `/trivia new {difficulty}`
   - Admin: `/trivia end` (ends current trivia)
   - Admin: `/leaderboard reset`
   - Non-admins receive error: "This command requires admin role"

## Accessibility

- Responses in Hungarian by default, English when user queries in English (automatic language detection)
- Clear error messages in appropriate language
- Simple, intuitive commands with help text

## Branding

- Bot name: "WH40K Lore Bot" (configured via .env)
- Avatar: Aquila symbol or WH40K themed icon
- Tone: Professional yet approachable, personality mode configured via .env
- Primary audience: Hungarian WH40K fans, with English support for international users

## Target Platforms

- Discord (desktop and mobile apps)
- Cross-platform (wherever Discord runs)

---
