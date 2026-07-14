from setuptools import setup


setup(
    name="tlux-search-hkm",
    version="0.1.0",
    description="Hierarchical token and semantic search index",
    packages=[
        "tlux.search.hkm",
        "tlux.search.hkm.builder",
        "tlux.search.hkm.search",
        "tlux.search.hkm.tools",
    ],
    package_dir={
        "tlux.search.hkm": ".",
        "tlux.search.hkm.builder": "builder",
        "tlux.search.hkm.search": "search",
        "tlux.search.hkm.tools": "tools",
    },
    install_requires=["numpy"],
    entry_points={"console_scripts": [
        "hkm-index=tlux.search.hkm.builder.launcher:main",
        "hkm-search=tlux.search.hkm.search.searcher:main",
        "hkm-audit=tlux.search.hkm.tools.audit:main",
        "hkm-inspect=tlux.search.hkm.tools.inspect:main",
        "hkm-benchmark=tlux.search.hkm.tools.benchmark:main",
        "hkm-agent=tlux.search.hkm.tools.local_agent:main",
        "hkm-language-benchmark=tlux.search.hkm.tools.language_benchmark:main",
    ]},
)
