from setuptools import setup

setup(
        name="tweets-sentiment",
        version="1.0",
        author="Novica Sarenac, Predrag Njegovanovic, Sasa Ferenc",
        author_email="novicasarenac@gmail.com, djaps94@gmail.com",
        description="Sentiment analysis of Twitter posts and comments",
        licence="MIT",
        url="https://github.com/novicasarenac/tweets-sentiment",
        packages=[
            "tweets-sentiment"
        ],
        install_requires=[
            "numpy",
            "scipy",
            "matplotlib"
        ]
    )
