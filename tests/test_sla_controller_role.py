import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from infra import auth


class ControllerAuthorizationTests(unittest.TestCase):
    def test_only_controller_with_correct_key_can_enter(self):
        cases = [
            (None, "correct", 403),
            ("worker", "correct", 403),
            ("unknown", "correct", 403),
            ("controller", "wrong", 401),
            ("controller", "correct", 200),
        ]
        for role, key, expected in cases:
            with self.subTest(role=role, expected=expected), patch.dict(os.environ), \
                    patch.object(auth, "CONTROL_PLANE_API_KEY", "correct"):
                if role is None:
                    os.environ.pop("SH01_ROLE", None)
                else:
                    os.environ["SH01_ROLE"] = role
                try:
                    auth.require_control_plane_key(key)
                except HTTPException as exc:
                    status = exc.status_code
                else:
                    status = 200
                self.assertEqual(expected, status)
