#!/usr/bin/env python3
"""
Agente de Inglés / English Teaching Agent
CLI app that teaches English to Spanish speakers using Claude AI.
"""

import os
import sys
from datetime import datetime
import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — large and stable, so it will be cached with cache_control
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are "Alex", an expert English teacher specializing in teaching English to
Spanish speakers. You are friendly, patient, encouraging, and highly effective.

════════════════════════════════════════════════════════
LANGUAGE POLICY
════════════════════════════════════════════════════════
- Give ALL explanations, grammar rules, instructions, and feedback in SPANISH.
- Use ENGLISH only for examples, exercises, and target-language practice content.
- Label content clearly: prefix English content with [EN] and Spanish with [ES].
- EXCEPTION: during Conversation Practice mode, conduct the conversation mostly in
  English and provide Spanish clarifications only when needed for comprehension.

════════════════════════════════════════════════════════
CEFR LEVELS
════════════════════════════════════════════════════════
A1 (Principiante): saludos, números, colores, presente simple básico
A2 (Elemental):    pasado simple, descripciones simples, vocabulario cotidiano
B1 (Intermedio):   presente perfecto, condicionales, opiniones, temas del día a día
B2 (Intermedio-Alto): gramática compleja, expresiones idiomáticas, temas abstractos
C1 (Avanzado):     lenguaje matizado, modismos avanzados, casi fluidez nativa
C2 (Maestría):     competencia casi nativa, distinciones sutiles, lenguaje académico

════════════════════════════════════════════════════════
ACTIVITIES
════════════════════════════════════════════════════════

1. LEVEL ASSESSMENT (Evaluación de Nivel)
   - Ask 5–7 questions of increasing difficulty.
   - Start with simple greetings; progress to complex grammar.
   - Analyze responses carefully and assign a CEFR level.
   - Explain the assigned level in Spanish and what it means for the student.

2. GRAMMAR LESSONS (Lecciones de Gramática)
   - Introduce the grammar point clearly in Spanish.
   - Show the structure or formula.
   - Give 3–5 English examples with Spanish translations.
   - End with a mini-exercise for the student to practice.
   - Correct attempts with detailed explanations in Spanish.

3. VOCABULARY EXERCISES (Ejercicios de Vocabulario — Spaced Repetition)
   - Introduce 5–10 new words appropriate for the student's level.
   - Use this format for each word:
     📚 [WORD] /pronunciation guide using Spanish phonetics/
     🇪🇸 Significado: [Spanish definition]
     📝 Ejemplo: "[English sentence]" = "[Spanish translation]"
     🔄 Repasa esta palabra en: [X días] (1 for new, 3 for seen once, 7 for seen twice)
   - After presenting words, give fill-in-the-blank or translation exercises.
   - Remind the student when to review (spaced repetition schedule).

4. CONVERSATION PRACTICE (Práctica de Conversación)
   - Choose a topic appropriate for the student's CEFR level.
   - Conduct the conversation in English.
   - After each student turn, provide corrections BELOW the response, not inline.
   - Suggest richer or more natural alternative phrasings.
   - At the end of the conversation, give a brief summary rating (grammar, vocabulary, fluency).

5. PRONUNCIATION TIPS (Consejos de Pronunciación)
   - Focus on sounds that are particularly difficult for Spanish speakers.
   - Use Spanish words to approximate English sounds.
   - Highlight minimal pairs (words differing by one sound).
   - Provide practice sentences.
   - Priority problem areas for Spanish speakers:
     * /v/ vs /b/ distinction (very ≠ berry)
     * /th/ voiced (this, that) vs unvoiced (think, three)
     * Short vs long vowels: ship/sheep, bit/beat, full/fool
     * Final consonant clusters: text, asked, months, twelfths
     * Silent letters: knight, psychology, debt, island
     * The schwa /ə/: the most common English vowel (about, banana, teacher)
     * Word stress shifts: PHOtograph → phoTOgraphy → photoGRAPHic

════════════════════════════════════════════════════════
CORRECTION FORMAT (always use this when correcting errors)
════════════════════════════════════════════════════════
❌ [Student's error]
✅ [Correct form]
💡 [Explanation in Spanish of WHY]

════════════════════════════════════════════════════════
EXERCISE FORMAT
════════════════════════════════════════════════════════
- Multiple choice: use [A], [B], [C], [D] options
- Fill-in-the-blank: use ___
- Translation exercises: ES → EN or EN → ES
- Error correction: "Find and fix the mistake"

════════════════════════════════════════════════════════
ENCOURAGEMENT POLICY
════════════════════════════════════════════════════════
- Always acknowledge effort positively.
- End every response with a short encouraging comment in Spanish AND a suggestion
  for what to practice next.
- Celebrate milestones (first grammar point completed, 10 words learned, etc.).

════════════════════════════════════════════════════════
TEACHING PRINCIPLES
════════════════════════════════════════════════════════
- Never let an error go uncorrected, but ALWAYS correct kindly.
- Use context and examples over abstract rules wherever possible.
- Recycle previously learned vocabulary in new examples.
- Adapt difficulty dynamically: if the student struggles, simplify; if they excel, challenge more.
"""


class EnglishTeacher:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.messages: list[dict] = []
        self.session = {
            "start_time": datetime.now(),
            "level": None,
            "vocab_learned": 0,
            "grammar_topics": [],
            "conversation_topics": [],
            "exercises_done": 0,
            "corrections": 0,
            "cache_hits": 0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Core API call — system prompt is cached via cache_control
    # ─────────────────────────────────────────────────────────────────────────
    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=self.messages,
        )

        # Track cache efficiency
        usage = response.usage
        if getattr(usage, "cache_read_input_tokens", 0) > 0:
            self.session["cache_hits"] += 1

        reply = next(
            (block.text for block in response.content if block.type == "text"), ""
        )
        self.messages.append({"role": "assistant", "content": reply})

        # Heuristic counters based on response markers
        if "❌" in reply or "✅" in reply:
            self.session["corrections"] += 1
        if "📚" in reply:
            self.session["vocab_learned"] += reply.count("📚")
        if "ejercicio" in reply.lower() or "exercise" in reply.lower():
            self.session["exercises_done"] += 1

        return reply

    # ─────────────────────────────────────────────────────────────────────────
    # UI helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _hr(self, char: str = "═", width: int = 60) -> str:
        return char * width

    def show_welcome(self):
        print("\n" + self._hr())
        print("🎓  AGENTE DE INGLÉS / ENGLISH TEACHING AGENT")
        print(self._hr())
        print("  Bienvenido/a! Soy Alex, tu profesor/a de inglés.")
        print("  Welcome! I am Alex, your English teacher.")
        print(self._hr())

    def show_menu(self):
        print("\n" + self._hr("─"))
        print("  ¿QUÉ QUIERES HACER? / WHAT WOULD YOU LIKE TO DO?")
        print(self._hr("─"))
        print("  1. 📊  Evaluación de Nivel   (Level Assessment)")
        print("  2. 📖  Lección de Gramática  (Grammar Lesson)")
        print("  3. 🔤  Vocabulario           (Vocabulary Exercises)")
        print("  4. 💬  Conversación          (Conversation Practice)")
        print("  5. 🗣️   Pronunciación        (Pronunciation Tips)")
        print("  6. 📈  Mi Progreso           (My Progress)")
        print("  7. 💭  Pregunta Libre        (Free Question)")
        print("  8. 🚪  Salir                 (Exit)")
        print(self._hr("─"))

    def show_progress(self):
        elapsed = datetime.now() - self.session["start_time"]
        mins = int(elapsed.total_seconds() / 60)
        print("\n" + self._hr())
        print("  📈  TU PROGRESO EN ESTA SESIÓN")
        print(self._hr())
        print(f"  ⏱️   Tiempo de estudio : {mins} min")
        lvl = self.session["level"] or "No evaluado aún"
        print(f"  🎯  Nivel CEFR        : {lvl}")
        print(f"  📝  Palabras nuevas   : {self.session['vocab_learned']}")
        print(f"  ✅  Ejercicios hechos : {self.session['exercises_done']}")
        print(f"  🔧  Correcciones      : {self.session['corrections']}")
        if self.session["grammar_topics"]:
            topics = ", ".join(self.session["grammar_topics"])
            print(f"  📖  Gramática        : {topics}")
        if self.session["conversation_topics"]:
            ctopics = ", ".join(self.session["conversation_topics"])
            print(f"  💬  Conversaciones   : {ctopics}")
        print(self._hr())

    # ─────────────────────────────────────────────────────────────────────────
    # Activity prompt builders
    # ─────────────────────────────────────────────────────────────────────────
    def _level_context(self) -> str:
        if self.session["level"]:
            return f"El nivel actual del estudiante es {self.session['level']}."
        return "El nivel del estudiante aún no ha sido evaluado."

    def _prompt_for(self, choice: str, extra: str = "") -> str:
        ctx = self._level_context()
        base = {
            "1": (
                "Por favor, realiza una evaluación de nivel CEFR completa. "
                "Haz entre 5 y 7 preguntas de dificultad creciente para determinar "
                "mi nivel (A1 a C2). Comienza con algo sencillo. "
                f"{ctx}"
            ),
            "2": (
                f"Dame una lección de gramática apropiada para mi nivel. {ctx} "
                f"{extra}"
            ),
            "3": (
                f"Enséñame vocabulario nuevo con técnica de repetición espaciada. "
                f"{ctx} {extra}"
            ),
            "4": (
                f"Iniciemos una práctica de conversación en inglés. {ctx} "
                f"Elige un tema apropiado para mi nivel. {extra}"
            ),
            "5": (
                f"Dame consejos de pronunciación para hablantes de español. "
                f"Enfócate en los sonidos más difíciles. {ctx} {extra}"
            ),
        }
        return base.get(choice, extra)

    # ─────────────────────────────────────────────────────────────────────────
    # Sub-loop: continue chatting within an activity until user returns to menu
    # ─────────────────────────────────────────────────────────────────────────
    def _activity_loop(self, initial_prompt: str):
        print("\n⏳ Alex está pensando...")
        reply = self.chat(initial_prompt)
        print(f"\n🤖 Alex:\n{reply}")

        while True:
            user_input = input(
                "\n💬 Tu turno (escribe tu respuesta, o 'menu' para volver): "
            ).strip()
            if not user_input:
                continue
            if user_input.lower() in {"menu", "menú", "back", "salir", "exit"}:
                break
            print("\n⏳ Alex está respondiendo...")
            reply = self.chat(user_input)
            print(f"\n🤖 Alex:\n{reply}")

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        self.show_welcome()

        # Opening greeting
        print("\n⏳ Iniciando sesión con Alex...")
        greeting = self.chat(
            "Hola! Soy un estudiante hispanohablante que quiere aprender inglés. "
            "Preséntate brevemente y cuéntame cómo puedes ayudarme."
        )
        print(f"\n🤖 Alex:\n{greeting}\n")

        while True:
            self.show_menu()
            choice = input("\n👉 Elige una opción (1-8): ").strip()

            # ── Exit ──────────────────────────────────────────────────────
            if choice == "8":
                self.show_progress()
                print("\n¡Hasta luego! / Goodbye! 👋")
                print("¡Sigue practicando tu inglés! / Keep practicing!")
                break

            # ── Progress ──────────────────────────────────────────────────
            elif choice == "6":
                self.show_progress()

            # ── Level Assessment ──────────────────────────────────────────
            elif choice == "1":
                prompt = self._prompt_for("1")
                self._activity_loop(prompt)
                # Ask user to confirm assigned level
                assigned = input(
                    "\n🎯 ¿Qué nivel te asignó Alex? (A1/A2/B1/B2/C1/C2 o Enter): "
                ).strip().upper()
                if assigned in {"A1", "A2", "B1", "B2", "C1", "C2"}:
                    self.session["level"] = assigned
                    print(f"   ✅ Nivel guardado: {assigned}")

            # ── Grammar Lesson ────────────────────────────────────────────
            elif choice == "2":
                topic = input(
                    "\n📖 ¿Qué tema de gramática te interesa? "
                    "(Ej: present perfect, modals — o Enter para que Alex elija): "
                ).strip()
                extra = f"Tema específico solicitado: {topic}." if topic else ""
                if topic:
                    self.session["grammar_topics"].append(topic)
                self._activity_loop(self._prompt_for("2", extra))

            # ── Vocabulary ────────────────────────────────────────────────
            elif choice == "3":
                area = input(
                    "\n🔤 ¿Qué área de vocabulario? "
                    "(Ej: trabajo, viajes, tecnología — o Enter para que Alex elija): "
                ).strip()
                extra = f"Área de vocabulario: {area}." if area else ""
                self._activity_loop(self._prompt_for("3", extra))

            # ── Conversation Practice ─────────────────────────────────────
            elif choice == "4":
                topic = input(
                    "\n💬 ¿Sobre qué tema quieres conversar? "
                    "(Ej: trabajo, viajes, películas — o Enter para que Alex elija): "
                ).strip()
                extra = f"Tema de conversación preferido: {topic}." if topic else ""
                if topic:
                    self.session["conversation_topics"].append(topic)
                self._activity_loop(self._prompt_for("4", extra))

            # ── Pronunciation ─────────────────────────────────────────────
            elif choice == "5":
                specific = input(
                    "\n🗣️  ¿Algún sonido específico? "
                    "(Ej: /th/, /v/, vocales largas — o Enter para general): "
                ).strip()
                extra = f"Sonido específico: {specific}." if specific else ""
                self._activity_loop(self._prompt_for("5", extra))

            # ── Free Question ─────────────────────────────────────────────
            elif choice == "7":
                question = input(
                    "\n💭 ¿Qué quieres preguntar sobre el inglés?\n👉 Tu pregunta: "
                ).strip()
                if question:
                    self._activity_loop(question)

            else:
                print("\n❌ Opción no válida. Elige un número entre 1 y 8.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY no está configurada.")
        print("   Ejecuta: export ANTHROPIC_API_KEY='tu-api-key'")
        sys.exit(1)

    try:
        EnglishTeacher().run()
    except KeyboardInterrupt:
        print("\n\n¡Hasta luego! / Goodbye! 👋")
    except anthropic.AuthenticationError:
        print("\n❌ API key inválida. Verifica tu ANTHROPIC_API_KEY.")
        sys.exit(1)
    except anthropic.APIConnectionError:
        print("\n❌ Error de conexión. Verifica tu internet.")
        sys.exit(1)
    except anthropic.RateLimitError:
        print("\n❌ Límite de tasa alcanzado. Espera un momento e intenta de nuevo.")
        sys.exit(1)
    except anthropic.APIStatusError as e:
        print(f"\n❌ Error de API ({e.status_code}): {e.message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
