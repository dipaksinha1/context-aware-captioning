from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from context_caption.schema import Project, Sound

APP = str(Path(__file__).parents[1] / "app.py")


class UITests(unittest.TestCase):
    def test_sample_edit_export_flow(self):
        app = AppTest.from_file(APP)
        app.run(timeout=30)
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label == "Load sample project").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["project"].captions), 4)
        first_name = next(t for t in app.text_input if t.label == "SPEAKER_00")
        first_name.set_value("Presenter")
        next(b for b in app.button if b.label == "Save speaker names").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["project"].speakers["SPEAKER_00"], "Presenter")
        next(b for b in app.button if b.label == "Save caption edits").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["project"].captions[0].original_text,
                         "Welcome to our captioning project.")
        self.assertEqual([m.value for m in app.metric if m.label == "Sound cues"], ["2"])
        self.assertIn("[doorbell rings]", [t.value for t in app.text])
        self.assertEqual([s.text for s in app.session_state["project"].sounds],
                         ["doorbell rings", "beeping"])
        next(c for c in app.checkbox if c.label == "Show background sound cues").uncheck().run()
        self.assertNotIn("[doorbell rings]", [t.value for t in app.text])

    def test_sounds_without_speech(self):
        app = AppTest.from_file(APP)
        app.session_state["project"] = Project(
            [], sounds=[Sound(1, 0.5, 2, "rain falling", "rain falling", .7)])
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertIn("[rain falling]", [t.value for t in app.text])
        next(b for b in app.button if b.label == "Save caption edits").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["project"].sounds), 1)
