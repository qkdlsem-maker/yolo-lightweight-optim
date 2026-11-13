Data access and input identity for the learning-rate follow-up
Checked 13 November 2026

The numerical results and plotting/verification code are public in the sibling
completed_followup_20261111 and publication_reproduction_20261112 folders.
This access supplement identifies the third-party inputs and records additional
byte checks. It does not report new experiments or grant a new project license.

BDD100K legacy annotations

The study reconstructed its labels from the legacy 2018-format JSON distributed
by SoleSensei, https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k.
The public metadata reported version 2, last updated 21 May 2019. On 13 November
2026, both recorded single-file download routes worked without credentials.
The uncompressed JSON bytes matched the study's pre-existing SHA-256 records:

train: c1998792e385fd79213187576ca07d5249563d0ab83ae976688ad60668baca9e
       1,451,202,234 bytes
val:   c835be652f0c002897be51e74deedd16a1dcee0b7230f19c0456fba772f694af
       208,257,864 bytes

SOURCE_ACCESS_VERIFICATION.json contains the exact download URLs and checks.
Verify the uncompressed JSON, because a provider may change its ZIP wrapper.
The current validation response is a ZIP; the original research download was
recorded as uncompressed JSON. Their JSON content is identical.

The original BDD documentation explicitly distinguishes 2018 labels from the
quality-checked Detection 2020 labels. These byte checks authenticate the
recorded mirror inputs, not an official-host archive or Detection 2020 identity.
Do not replace these annotations with Detection 2020 and call it exact replay.
https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst

BDD images and reconstructed labels

The earlier YOLO distribution was traced to
https://www.kaggle.com/datasets/a7madmostafa/bdd100k-yolo (version 4) using five
matching samples. That tracing did not authenticate the complete distribution.
The current study uses 69,863 retained training images and 10,000 validation
images. It excludes the 137 training images with unauthenticated labels.
The source reconstruction contains 1,286,871 training label rows; the native
training loader uses 1,286,870 boxes after its duplicate handling. These counts
describe different stages and must not be interchanged.

Use the frozen clean_data_manifest.json, excluded-image list, ordered-label
digests and evaluation image hashes. Original preparation code is at:
experiments/audit_legacy_conversion.py
experiments/clean_followup/setup_clean_replication.py
artifacts/clean_followup_20260929/clean_data_manifest.json
artifacts/ieee_access_20260929/DATA_PROVENANCE.md

These scripts retain historical directory assumptions and should be reviewed
when relocating the project. A moved data YAML has a different file hash even
if image and label bytes are unchanged. Do not claim original-byte execution
proofs for newly adapted paths. The public annotation check is not a fresh
download-and-hash audit of all training images.

The official documentation points to https://dl.cv.ethz.ch/bdd100k/data/.
That host could not be resolved from the audit environment on 13 November 2026.
This is a dated access observation, not evidence that the service has ceased.

KITTI

Obtain the object-detection left color images and training labels through:
https://www.cvlibs.net/datasets/kitti/eval_object.php
The page routed its download links to provider login on 13 November 2026.
Follow the provider's registration and use conditions; no special access from
the study authors is offered or required. This study evaluates the 7,481
labeled training frames as an external evaluation set, not the hidden test set
or a leaderboard submission. It does not train on KITTI.

The recorded data_object_label_2.zip SHA-256 is:
4efc76220d867e1c31bb980bbf8cbc02599f02a9cb4350effa98dbb04aaed880
The independently verified study records bind original member bytes and image
identities. This access check did not create a provider account or re-download
the full KITTI image archive. KITTI's own page supplies registration/contact
routes for other researchers.

Third-party terms and remaining release status

Read THIRD_PARTY_NOTICES.txt before reuse. The BDD dataset license is separate
from the BSD license of BDD's code repository. KITTI uses CC BY-NC-SA 3.0.
Ultralytics v8.2.103 carries AGPL-3.0. A Kaggle mirror's license label does not
replace upstream rights. Original images and annotations are not redistributed
by this supplement.

ENDPOINT_ARTIFACT_BINDINGS.json resolves all 84 inference checkpoint digests by
following the frozen evidence INDEX to each independent evaluation audit,
ENDPOINT_COMPLETE.json and native/prediction manifests. Four frozen paths were
superseded by the recorded restart028 routes; completed manifests identify the
actual files. Eighteen reused continuation endpoints retain historical IDs
without the learning-rate suffix. No scientific record has been rewritten.

Model weights, raw training states, and native per-detection statistics are
being prepared separately for an underlying-output deposit. A list of hashes
is not the binary deposit. The old September 2026 GitHub Release does not cover
all 84 follow-up endpoints. Cached individual predictions remain distinct from
the already published numerical scores and from native validation statistics.

Author/rightsholder confirmation of project-code and model reuse terms remains
pending. No blanket CC BY, MIT or AGPL grant for author-owned work is made here.
The remaining output deposit and matching final manuscript/portal availability
statement must be completed before asserting full input/output availability.
PLOS decides whether the final access arrangements satisfy its policies:
https://journals.plos.org/plosone/s/data-availability
https://journals.plos.org/plosone/s/materials-software-and-code-sharing

Checking the recorded annotation download routes

Run python verify_source_access.py from a writable copy of this folder.
It downloads the two source archives into public_source_verification and checks
the uncompressed JSON hashes. It does not run training or inference. Allow
approximately 110 MB of network/disk space for the current ZIP responses;
provider packaging may change. Raw source data are not committed here.
The one-image access check in IMAGE_ACCESS_SAMPLE.json matched the recorded
validation-image bytes at both mirrors, without authentication. This sample
does not authenticate the complete image distributions.
