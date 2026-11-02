"""Commons metadata URL selection; no network or GPU required."""
import unittest
from ai_compose_commons import media_download_url, parse_candidates


class CommonsUrlTests(unittest.TestCase):
    def setUp(self):
        self.original = 'https://upload.wikimedia.org/wikipedia/commons/a/ab/Example.jpg'
        self.thumbnail = 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Example.jpg/1280px-Example.jpg'
        self.info = {'url': self.original, 'thumburl': self.thumbnail, 'mime': 'image/jpeg',
                     'size': 1000000, 'width': 1920, 'height': 1080,
                     'extmetadata': {'LicenseShortName': {'value': 'CC BY-SA 4.0'}}}

    def payload(self):
        return {'query': {'pages': {'1': {'pageid': 1, 'title': 'File:Example.jpg',
                                         'imageinfo': [self.info]}}}}

    def test_usable_thumbnail_still_preferred(self):
        self.assertEqual(media_download_url(self.info, 'image'), self.thumbnail)

    def test_rejected_thumbnail_falls_back_to_original(self):
        self.info['thumburl'] = 'https://commons.wikimedia.org/w/thumb.php?f=Example.jpg'
        result = parse_candidates(self.payload(), 'image')
        self.assertEqual(result[0]['download_url'], self.original)
        self.assertEqual(result[0]['license'], 'CC BY-SA 4.0')

    def test_protocol_relative_thumbnail_uses_https(self):
        self.info['thumburl'] = self.thumbnail.removeprefix('https:')
        self.assertEqual(media_download_url(self.info, 'image'), self.thumbnail)

    def test_malformed_thumbnail_falls_back(self):
        for value in ('https://[bad', 'https://upload.wikimedia.org:bad/file', None, 123):
            self.info['thumburl'] = value
            self.assertEqual(media_download_url(self.info, 'image'), self.original)

    def test_untrusted_hosts_credentials_and_ports_remain_rejected(self):
        for value in ('https://upload.wikimedia.org.evil.test/file',
                      'https://upload.wikimedia.org@evil.test/file',
                      'https://user:password@upload.wikimedia.org/file',
                      'https://upload.wikimedia.org:8443/file',
                      'http://upload.wikimedia.org/file'):
            self.assertIsNone(media_download_url({'thumburl': value, 'url': value}, 'image'))

    def test_video_uses_original_not_image_thumbnail(self):
        self.assertEqual(media_download_url(self.info, 'video'), self.original)

    def test_original_fallback_does_not_bypass_size_or_license_filters(self):
        self.info['thumburl'] = 'https://other.example/file.jpg'
        self.info['size'] = 81 * 1024 * 1024
        self.assertEqual(parse_candidates(self.payload(), 'image'), [])
        self.info['size'] = 1000
        self.info['extmetadata']['LicenseShortName']['value'] = 'All rights reserved'
        self.assertEqual(parse_candidates(self.payload(), 'image'), [])


if __name__ == '__main__':
    unittest.main()
