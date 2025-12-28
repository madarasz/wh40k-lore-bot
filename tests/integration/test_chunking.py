"""Integration tests for text chunking with real wiki articles."""

import pytest

from src.ingestion.text_chunker import MAX_TOKENS, MIN_TOKENS, MarkdownChunker


class TestChunkingIntegration:
    """Integration tests for chunking real wiki articles."""

    @pytest.fixture
    def chunker(self) -> MarkdownChunker:
        """Create a MarkdownChunker instance for testing."""
        return MarkdownChunker()

    @pytest.fixture
    def tyranids_article(self) -> str:  # noqa: PLR0915
        """Sample Tyranids article (realistic wiki structure, ~72KB).

        This simulates a real wiki article with multiple sections,
        subsections, and substantial content.
        """
        # Generate a realistic article structure with enough content
        # Long lines in markdown test data are expected
        # ruff: noqa: E501
        content = """## Overview
The Tyranids are an extragalactic alien race whose sole purpose is the consumption of all forms of genetic and biological material in order to evolve and reproduce. Tyranid technology is based entirely on biological engineering, every function and tool used by the Tyranids is a living organism that has been created, or rather evolved, to perform that specific task. The Tyranids are likened to a galactic swarm consuming everything in their path, feeding on entire worlds and draining them of all biological, organic and liquid resources. Once consumed, the biomass is transformed into more Tyranids which allows for their rapid growth and reproduction.

### The Hive Mind
The Hive Mind is the gestalt collective consciousness of the entire Tyranid race. It is a nearly omniscient entity composed of pure psychic energy that binds every Tyranid creature in a form of synaptic web. Every Tyranid creature from the lowliest Ripper to the mightiest Hive Tyrant is linked into a vast neural network. The Hive Mind is the ultimate expression of the Tyranid collective consciousness, an entity of such psychic might that it can enforce its will across light years of space. The presence of the Hive Mind is so powerful that it can drive lesser races insane, and even the most stalwart defenders of the Imperium have been known to falter in the face of its psychic onslaught.

The Hive Mind learns from every encounter, adapting and evolving its swarms to counter the tactics and weapons of its foes. This adaptive evolution makes the Tyranids incredibly dangerous, as they can develop resistances to weapons that were effective against them in previous encounters. The Hive Mind never forgets, and every adaptation learned by one hive fleet is eventually shared with others through mysterious means.

## History
### First Contact
The Tyranids were first encountered by the Imperium of Man in the Eastern Fringe of the galaxy. The first recorded contact was with Hive Fleet Behemoth in 741.M41. The arrival of the Tyranids was preceded by a psychic phenomenon known as the Shadow in the Warp, which disrupted all psychic communications and travel in the region. This shadow is a side effect of the Hive Mind's overwhelming psychic presence and serves to isolate their prey from reinforcement and aid.

The Ultramarines Chapter bore the brunt of the initial invasion, with many worlds in Ultramar being consumed before the threat was finally contained at the Battle of Macragge. Though Behemoth was eventually defeated, it was clear that this was merely the vanguard of a much larger threat. The Imperium soon learned that the Tyranids were not a single fleet, but rather multiple hive fleets approaching the galaxy from different vectors, each one composed of billions upon billions of bioforms.

### Major Hive Fleets
Since the first encounter, numerous hive fleets have been identified by the Imperium. Each fleet is a distinct invasion force, containing trillions of individual organisms, all controlled by the Hive Mind. The largest and most notable hive fleets include Behemoth, Kraken, and Leviathan. Each of these fleets has consumed countless worlds and killed billions of the Emperor's subjects. The Tyranids show no signs of stopping, and many in the Imperium fear that the fleets identified so far are merely the leading tendrils of a much larger swarm yet to arrive.

Hive Fleet Behemoth was the first major incursion, arriving from beneath the galactic plane in the Eastern Fringe. It was eventually halted at Macragge, but not before consuming dozens of worlds. Hive Fleet Kraken followed several decades later, splitting into multiple tendrils and attacking from numerous directions simultaneously, making it far more difficult to contain. Hive Fleet Leviathan is the most recent and largest threat, approaching from below the galactic plane and threatening to consume the entire Eastern Fringe.

## Biology and Organisms
### Synapse Creatures
Synapse creatures are the most important battlefield organisms in the Tyranid swarm. These creatures serve as nodes in the Hive Mind's neural network, relaying the will of the Hive Mind to lesser organisms and maintaining control over the swarm. Without synapse creatures, lesser Tyranids become little more than mindless beasts, prone to feeding frenzies and lacking any tactical coordination. The most common synapse creatures include Hive Tyrants, Warriors, and Tyranid Primes.

The presence of synapse creatures on the battlefield creates a synaptic web that extends the Hive Mind's control. Lesser organisms within this web benefit from enhanced coordination, fearlessness, and purpose-driven aggression. When synapse creatures are killed, the web weakens, and lesser organisms may fall back on their instinctive behaviors, which usually involves simply consuming any biomass available, friend or foe.

### Warrior Organisms
Tyranid Warriors are the most common synapse creatures and serve as the basic command structure for Tyranid swarms. Standing taller than a man and armed with a variety of bio-weapons, Warriors are formidable opponents in their own right. They often lead broods of lesser organisms such as Hormagaunts and Termagants, directing their actions with telepathic commands. Warriors are also highly adaptable, capable of being equipped with different weapons and biomorphs depending on the tactical situation.

Warriors are born with a fragment of the Hive Mind's intelligence, giving them a degree of autonomy while still remaining completely loyal to the swarm. This intelligence makes them dangerous opponents, capable of tactical thinking and coordinating complex attacks. In larger swarms, Warriors often serve as lieutenants to more powerful synapse creatures like Hive Tyrants.

### Monstrous Creatures
The Tyranid swarm includes numerous massive organisms bred for specific battlefield roles. These monstrous creatures are living engines of destruction, each one capable of turning the tide of battle. The Carnifex is perhaps the most infamous, a living battering ram equipped with massive claws and bio-cannons. Carnifexes are used to break through enemy fortifications and annihilate heavy infantry and vehicles.

Other monstrous creatures include the Tyrannofex, a living artillery platform armed with rupture cannons and acid sprays, and the Trygon, a massive serpentine creature that burrows through the ground to emerge in the midst of enemy lines. Each of these creatures represents a significant investment of biomass by the Hive Mind, but their battlefield impact often justifies the cost.

### Genestealer Cults
Genestealers represent a unique form of Tyranid infiltration. Unlike the majority of Tyranid organisms that arrive with the hive fleets, Genestealers infiltrate worlds years or even decades before the fleet's arrival. They infect human populations, creating hybrid cults that worship the Tyranids as gods. These cults slowly spread through the population, remaining hidden until the hive fleet arrives.

When the fleet enters the system, Genestealer Cults rise up in rebellion, sabotaging defenses and sowing chaos from within. This internal uprising often proves more devastating than the external invasion, as it strikes when the defenders are least prepared. The cult's psychic beacon also guides the hive fleet directly to the planet, ensuring that no world with an established cult can escape consumption.

## Combat Doctrine
### Swarm Tactics
The Tyranids employ overwhelming numbers as their primary tactical doctrine. Swarms of lesser organisms such as Hormagaunts and Termagants advance in vast waves, absorbing enemy firepower while more dangerous organisms move into position. This tactic, known as the Endless Swarm, has broken the will of countless defenders who find themselves drowning in a tide of chitin and claws.

The swarm is not merely mindless, however. The Hive Mind coordinates the assault with perfect precision, exploiting weaknesses and adapting to enemy tactics in real-time. Lesser organisms are sacrificed without hesitation to create opportunities for more valuable bioforms. This ruthless efficiency makes the Tyranids terrifying opponents, as they show no fear, take no prisoners, and never retreat except as a deliberate tactical maneuver.

### Adaptive Evolution
One of the most terrifying aspects of the Tyranid threat is their ability to adapt and evolve in response to enemy tactics. Between invasions, the Hive Mind processes the genetic material consumed from defeated foes and environments, incorporating useful traits into new organisms. This means that tactics which worked against one swarm may prove ineffective against the next, as the Tyranids have already evolved countermeasures.

This adaptation occurs on multiple timescales. Within a single battle, synapse creatures can direct the swarm to adjust tactics based on enemy behavior. Between battles, new organisms can be spawned with specific biomorphs designed to counter known threats. On a larger scale, entire hive fleets evolve new strains of organisms based on the accumulated experience of previous invasions.

## Notable Encounters
### The Battle of Macragge
The Battle of Macragge was the first major confrontation between the Imperium and a Tyranid hive fleet. Hive Fleet Behemoth descended upon Ultramar, consuming world after world until it reached Macragge itself, the realm of the Ultramarines. The battle was fierce and desperate, with the Ultramarines suffering devastating losses. The Chapter Master Marneus Calgar himself was gravely wounded in the fighting, his body broken by a Carnifex.

The tide turned when the Ultramarines fleet, led by Calgar despite his injuries, boarded the hive ships in orbit and destroyed them from within. The loss of the hive fleet's synapse nodes threw the ground forces into disarray, allowing the Ultramarines to finally drive back the swarm. Though victorious, Ultramar was left scarred, and the Ultramarines had learned a terrible lesson about the true nature of the Tyranid threat.

### The Octarius War
The Octarius War represents one of the most catastrophic decisions in Imperial history. Inquisitor Kryptman, seeking to slow Hive Fleet Leviathan, deliberately directed the hive fleet toward the Ork empire of Octarius. His plan was for the two xenos races to destroy each other, buying the Imperium precious time. While the initial stages went as planned, with the Tyranids and Orks locked in eternal combat, the long-term consequences may prove disastrous.

The war has continued for decades, with neither side gaining a decisive advantage. More worryingly, the Tyranids are rapidly evolving in response to Ork physiology and tactics, incorporating Ork genetic material into new strains. The Orks, for their part, are growing stronger and more numerous due to the constant warfare. Many fear that the victor of the Octarius War, whether Tyranid or Ork, will emerge stronger than either was before, posing an even greater threat to the Imperium.

## Strategic Threat Assessment
The Tyranid threat represents an existential danger to all life in the galaxy. Unlike other xenos races which can be negotiated with, contained, or at least understood, the Tyranids are driven by pure biological imperative. They cannot be reasoned with, they cannot be bought off, and they will never stop consuming until all biomass in the galaxy has been devoured. Each world they consume makes them stronger, and each battle teaches them how to fight more effectively.

The full extent of the Tyranid invasion is unknown. The hive fleets identified so far may represent only a fraction of the total swarm. Some xenobiologists theorize that the Tyranids have already consumed entire galaxies, and that the Milky Way is merely their next feeding ground. If true, the Imperium faces a threat that cannot be defeated, only delayed. The best humanity can hope for is to make the consumption of each world so costly that the Tyranids eventually decide to seek easier prey elsewhere, though there is no evidence that such calculations factor into the Hive Mind's decision-making.

The Shadow in the Warp that precedes each hive fleet makes coordinated defense nearly impossible. Astropathic communication is disrupted, preventing worlds from calling for aid and fleets from coordinating their movements. This isolation is often as devastating as the invasion itself, as worlds must face the Tyranid swarm alone, without hope of reinforcement. Only through constant vigilance, technological innovation, and the sacrifice of countless lives can the Imperium hope to slow the Tyranid advance and buy time for the future generations who will continue this endless war.
"""
        return content

    def test_tyranids_chunking(self, chunker: MarkdownChunker, tyranids_article: str) -> None:
        """Test chunking a large realistic article.

        Verifies:
        - All chunks within token limits
        - Section paths preserved correctly
        - No content lost
        - Chunking statistics logged
        """
        chunks = chunker.chunk_markdown(tyranids_article, "Tyranids")

        # Should produce multiple chunks
        assert len(chunks) > 1, "Article should be split into multiple chunks"

        # Verify all chunks within token limits
        for i, chunk in enumerate(chunks):
            token_count = chunker._count_tokens(chunk.chunk_text)
            assert token_count >= MIN_TOKENS or i == len(chunks) - 1, (
                f"Chunk {i} has {token_count} tokens, below {MIN_TOKENS} "
                "(only last chunk can be below minimum)"
            )
            assert token_count <= MAX_TOKENS, (
                f"Chunk {i} has {token_count} tokens, exceeds {MAX_TOKENS}"
            )

        # Verify section paths are correct
        section_paths = [chunk.section_path for chunk in chunks]
        expected_sections = [
            "Overview",
            "Overview > The Hive Mind",
            "History",
            "History > First Contact",
            "History > Major Hive Fleets",
            "Biology and Organisms",
            "Biology and Organisms > Synapse Creatures",
            "Biology and Organisms > Warrior Organisms",
            "Biology and Organisms > Monstrous Creatures",
            "Biology and Organisms > Genestealer Cults",
            "Combat Doctrine",
            "Combat Doctrine > Swarm Tactics",
            "Combat Doctrine > Adaptive Evolution",
            "Notable Encounters",
            "Notable Encounters > The Battle of Macragge",
            "Notable Encounters > The Octarius War",
            "Strategic Threat Assessment",
        ]

        # Verify all expected sections appear in chunks
        for expected in expected_sections:
            assert any(expected in path for path in section_paths), (
                f"Expected section '{expected}' not found in chunks"
            )

        # Verify all chunks have required fields
        for chunk in chunks:
            assert chunk.chunk_text, "Chunk text should not be empty"
            assert chunk.article_title == "Tyranids", "Article title should be 'Tyranids'"
            assert chunk.section_path, "Section path should not be empty"
            assert isinstance(chunk.chunk_index, int), "Chunk index should be integer"
            assert chunk.chunk_index >= 0, "Chunk index should be non-negative"

        # Verify chunk indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i, f"Chunk index should be {i}, got {chunk.chunk_index}"

        # Calculate and log statistics
        token_counts = [chunker._count_tokens(chunk.chunk_text) for chunk in chunks]
        total_tokens = sum(token_counts)
        avg_tokens = total_tokens // len(chunks)
        min_chunk_tokens = min(token_counts)
        max_chunk_tokens = max(token_counts)

        print("\n--- Tyranids Article Chunking Statistics ---")
        print(f"Total chunks: {len(chunks)}")
        print(f"Total tokens: {total_tokens}")
        print(f"Average tokens per chunk: {avg_tokens}")
        print(f"Min chunk size: {min_chunk_tokens} tokens")
        print(f"Max chunk size: {max_chunk_tokens} tokens")
        print(f"Unique section paths: {len(set(section_paths))}")

        # Verify no content lost by checking total character count
        original_chars = len(tyranids_article)
        chunked_chars = sum(len(chunk.chunk_text) for chunk in chunks)

        # Due to header inclusion in chunks, chunked_chars may be slightly larger
        # but should be in the same ballpark
        char_ratio = chunked_chars / original_chars
        assert 0.95 <= char_ratio <= 1.2, (
            f"Character count mismatch: original={original_chars}, chunked={chunked_chars}"
        )

    def test_small_article(self, chunker: MarkdownChunker) -> None:
        """Test chunking a small article that doesn't need splitting."""
        small_article = """## Overview
This is a small article about a minor topic.
It only has one section and minimal content."""

        chunks = chunker.chunk_markdown(small_article, "Small Article")

        assert len(chunks) == 1
        assert chunks[0].section_path == "Overview"
        assert chunks[0].chunk_index == 0

    def test_article_with_deep_nesting(self, chunker: MarkdownChunker) -> None:
        """Test article with deeply nested sections.

        Note: Very small sections may be merged together, so we verify that
        hierarchical structure is parsed correctly rather than expecting
        every tiny section to be a separate chunk.
        """
        nested_article = """## Chapter One
This is chapter one with some substantial content to ensure it's not too small.
The chapter discusses important topics related to the lore and history.

### Section A
This is section A under chapter one with detailed information about the topic.
It contains multiple sentences to ensure sufficient token count for testing.

#### Subsection A1
This is subsection A1 with comprehensive details and extensive coverage of subtopics.
Additional content ensures this section has enough tokens to be meaningful.

#### Subsection A2
This is subsection A2 with its own detailed content and information.
More text to ensure proper token count and testing coverage.

### Section B
This is section B under chapter one with substantial discussion of the subject matter.
Multiple paragraphs ensure this section is properly sized for chunking.

## Chapter Two
This is chapter two with its own comprehensive coverage of different topics.
Substantial content ensures proper testing of the chunking algorithm.

### Section C
This is section C under chapter two with detailed explanations and examples.
Additional content provides sufficient tokens for meaningful chunk creation."""

        chunks = chunker.chunk_markdown(nested_article, "Nested Article")

        # Verify at least some hierarchical structure is preserved
        paths = [chunk.section_path for chunk in chunks]
        # Check that we have some hierarchical paths (not just merged everything)
        assert any(">" in path for path in paths), "Should have some hierarchical paths"

        # Verify major sections appear
        path_str = " ".join(paths)
        assert "Chapter One" in path_str or "Section A" in path_str
        assert "Chapter Two" in path_str or "Section C" in path_str
